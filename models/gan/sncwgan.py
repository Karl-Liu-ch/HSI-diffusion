import sys
sys.path.append('./')
import torch.nn as nn
import torch
# torch.manual_seed(1234)
from torch.nn import functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.optim as optim
import torch.backends.cudnn as cudnn
from options import opt
import os
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import autograd
import functools
from models.gan.networks import *
from models.transformer.DTN import DTN
import scipy.io
import numpy as np
import re
from dataset.datasets import TestFullDataset
from models.gan.Basemodel import BaseModel, criterion_mrae, AverageMeter, SAM
from utils import AverageMeter, record_loss, Loss_MRAE, Loss_RMSE, Loss_PSNR, Loss_Fid, \
    Loss_SAM, Loss_SSIM, reconRGB, Loss_SID, SAM, SaveSpectral
from models.gan.Utils import Log_loss, Itself_loss
from models.transformer.MST_Plus_Plus import MST_Plus_Plus
from color_loss import deltaELoss
import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
from utils import instantiate_from_config
from omegaconf import OmegaConf
config_path = "configs/sncwgan_dtn.yaml"
cfg = OmegaConf.load(config_path)

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
if opt.multigpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion_mrae = Loss_MRAE()
criterion_rmse = Loss_RMSE()
criterion_psnr = Loss_PSNR()
criterion_psnrrgb = Loss_PSNR()
criterion_sam = Loss_SAM()
criterion_sid = Loss_SID()
criterion_fid = Loss_Fid().cuda()
criterion_ssim = Loss_SSIM().cuda()

class SpectralNormalizationCGAN(pl.LightningModule):
    def __init__(self,
                 modelconfig,
                 lossconfig, 
                 epochs,
                 learning_rate,
                 cond_key, 
                 ):
        super().__init__()
        self.cond_key = cond_key
        self.automatic_optimization = False
        self.end_epoch = epochs
        self.lr = learning_rate
        self.generator = instantiate_from_config(modelconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.root = f'/work3/s212645/Spectral_Reconstruction/checkpoint/SNCWGAN/DTN_{lossconfig.params.l1_weight}_{lossconfig.params.sam_weight}/'
        self.init_metrics()
    
    def init_metrics(self):
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        self.metrics = {
            'MRAE':np.zeros(shape=[self.end_epoch]),
            'RMSE':np.zeros(shape=[self.end_epoch]),
            'PSNR':np.zeros(shape=[self.end_epoch]),
            'SAM':np.zeros(shape=[self.end_epoch])
        }

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")
    
    def forward(self, input):
        reconstructions = self.generator(input)
        return reconstructions

    def training_step(self, batch, batch_idx):
        opt_g, opt_disc = self.optimizers()
        cond = batch[self.cond_key]
        labels = batch['label']
        cond = Variable(cond)
        labels = Variable(labels)
        rec = self(cond)

        # generator
        self.toggle_optimizer(opt_g)
        g_loss, log_dict_g = self.loss(rec, labels, cond, 0, split="train")
        self.log("train/aeloss", g_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_g, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.manual_backward(g_loss)
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)

        # discriminator
        self.toggle_optimizer(opt_disc)
        discloss, log_dict_disc = self.loss(rec, labels, cond, 1, split="train")
        self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.manual_backward(discloss)
        opt_disc.step()
        opt_disc.zero_grad()
        self.untoggle_optimizer(opt_disc)

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.losses_mrae = AverageMeter()
            self.losses_rmse = AverageMeter()
            self.losses_psnr = AverageMeter()
            self.losses_sam = AverageMeter()
        images = batch[self.cond_key]
        labels = batch['label']
        images = Variable(images)
        labels = Variable(labels)
        output = self.generator(images)
        loss_mrae = criterion_mrae(output, labels).detach()
        loss_rmse = criterion_rmse(output, labels).detach()
        loss_psnr = criterion_psnr(output, labels).detach()
        loss_sam = criterion_sam(output, labels).detach()
        criterion_sam.reset()
        criterion_psnr.reset()
        self.log_dict({'mrae': loss_mrae, 'rmse': loss_rmse, 'psnr': loss_psnr, 'sam': loss_sam})
        self.losses_mrae.update(loss_mrae.data)
        self.losses_rmse.update(loss_rmse.data)
        self.losses_psnr.update(loss_psnr.data)
        self.losses_sam.update(loss_sam.data)
        # if batch_idx == 0:
        #     print(f'validation: MRAE: {self.losses_mrae.avg}, RMSE: {self.losses_rmse.avg}, PSNR: {self.losses_psnr.avg}, SAM: {self.losses_sam.avg}.')
        return self.log_dict
    
    def on_validation_epoch_end(self):
        print(f'validation: MRAE: {self.losses_mrae.avg}, RMSE: {self.losses_rmse.avg}, PSNR: {self.losses_psnr.avg}, SAM: {self.losses_sam.avg}.')
        self.log_dict({'mrae': self.losses_mrae.avg, 'rmse': self.losses_rmse.avg, 'psnr': self.losses_psnr.avg, 'sam': self.losses_sam.avg})
    
    def configure_optimizers(self):
        lr = self.lr
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_g, opt_disc], []

    # def training_step(self, images, labels):
    #     self.optimG.zero_grad()
    #     outputs = self.G(images)
    #     loss1 = deltaELoss(outputs, labels)
    #     loss1.backward()

    #     gradients1 = [param.grad.clone() for param in self.G.parameters()]

    #     self.optimG.zero_grad()
    #     outputs = self.G(images)
    #     loss2 = self.lossl1(outputs, labels)
    #     loss2.backward()

    #     mask_gradients = []
    #     for grad1, grad2 in zip(gradients1, self.G.parameters()):
    #         mask1 = torch.abs(torch.sign(grad1) + torch.sign(grad2.grad)) / 2.0
    #         mask2 = - torch.abs(torch.sign(grad1) - torch.sign(grad2.grad)) / 2.0
    #         mask = mask1 + mask2
    #         mask_gradients.append(grad2.grad + mask * grad1)

    #     for param, grad in zip(self.G.parameters(), mask_gradients):
    #         param.grad = grad

    #     self.optimG.step()

    #     return loss2, loss1

    # def train(self):
    #     super().train()
    #     while self.epoch<self.end_epoch:
    #         self.G.train()
    #         self.D.train()
    #         losses = AverageMeter()
    #         train_loader = DataLoader(dataset=self.train_data, batch_size=self.opt.batch_size, shuffle=True, num_workers=2,
    #                                 pin_memory=True, drop_last=True)
    #         val_loader = DataLoader(dataset=self.val_data, batch_size=self.opt.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    #         for i, (_, labels, images) in enumerate(train_loader):
    #             labels = labels.cuda()
    #             images = images.cuda()
    #             images = Variable(images)
    #             labels = Variable(labels)
    #             if self.nonoise:
    #                 z = images
    #             else:
    #                 z = torch.randn_like(images).cuda()
    #                 z = torch.concat([z, images], dim=1)
    #                 z = Variable(z)
    #             realAB = torch.concat([images, labels], dim=1)
    #             # D_real, D_real_feature = self.D(realAB)
    #             x_fake = self.G(z)
    #             fakeAB = torch.concat([images, x_fake],dim=1)
                
    #             # train D
    #             for p in self.D.parameters():
    #                 p.requires_grad = True
    #             self.optimD.zero_grad()
    #             D_real, D_real_feature = self.D(realAB)
    #             loss_real = -D_real.mean(0).view(1)
    #             loss_real.backward(retain_graph = True)
    #             D_fake, _ = self.D(fakeAB.detach())
    #             loss_fake = D_fake.mean(0).view(1)
    #             loss_fake.backward()
    #             self.optimD.step()
                
    #             # train G
    #             self.training_step(images, labels)
    #             self.optimG.zero_grad()
    #             lrG = self.optimG.param_groups[0]['lr']
    #             for p in self.D.parameters():
    #                 p.requires_grad = False
    #             pred_fake, D_fake_feature = self.D(fakeAB)
    #             loss_G = -pred_fake.mean(0).view(1)
    #             lossl1 = self.lossl1(x_fake, labels) * self.lamda
    #             losssam = SAM(x_fake, labels) * self.lambdasam
    #             perceptual_loss = 0
    #             for k in range(len(D_fake_feature)):
    #                 perceptual_loss += nn.MSELoss()(D_real_feature[k].detach(), D_fake_feature[k])
    #             loss_G += lossl1 + losssam + perceptual_loss * self.lambdaperceptual
    #             # train the generator
    #             loss_G.backward()
    #             self.optimG.step()
                
    #             loss_mrae = criterion_mrae(x_fake, labels)
    #             losses.update(loss_mrae.data)
    #             self.iteration = self.iteration+1
    #             if self.iteration % 20 == 0:
    #                 print('[iter:%d/%d],lr=%.9f,train_losses.avg=%.9f'
    #                     % (self.epoch, self.end_epoch, lrG, losses.avg))
    #         # validation
    #         mrae_loss, rmse_loss, psnr_loss, sam_loss, sid_loss = self.validate(val_loader)
    #         print(f'MRAE:{mrae_loss}, RMSE: {rmse_loss}, PNSR:{psnr_loss}, SAM: {sam_loss}, SID: {sid_loss}')
    #         # Save model
    #         # if torch.abs(mrae_loss - record_mrae_loss) < 0.01 or mrae_loss < record_mrae_loss or self.iteration % 5000 == 0:
    #         print(f'Saving to {self.root}')
    #         self.save_checkpoint()
    #         if mrae_loss < self.best_mrae:
    #             self.best_mrae = mrae_loss
    #             self.save_checkpoint(True)
    #         # print loss
    #         print(" Iter[%06d], Epoch[%06d], learning rate : %.9f, Train MRAE: %.9f, Test MRAE: %.9f, "
    #             "Test RMSE: %.9f, Test PSNR: %.9f, SAM: %.9f, SID: %.9f " % (self.iteration, 
    #                                                             self.epoch, lrG, 
    #                                                             losses.avg, mrae_loss, rmse_loss, psnr_loss, sam_loss, sid_loss))
    #         self.epoch += 1
    #         self.schedulerD.step()
    #         self.schedulerG.step()

    # def test_full_resol(self, modelname):
    #     try:
    #         os.mkdir('/work3/s212645/Spectral_Reconstruction/FakeHyperSpectrum/' + modelname + '/FullResol/')
    #     except:
    #         pass
    #     root = '/work3/s212645/Spectral_Reconstruction/RealHyperSpectrum/'
    #     H_ = 128
    #     W_ = 128
    #     losses_mrae = AverageMeter()
    #     losses_rmse = AverageMeter()
    #     losses_psnr = AverageMeter()
    #     losses_psnrrgb = AverageMeter()
    #     losses_sam = AverageMeter()
    #     losses_sid = AverageMeter()
    #     losses_fid = AverageMeter()
    #     losses_ssim = AverageMeter()
    #     test_data = TestFullDataset(data_root=opt.data_root, crop_size=opt.patch_size, valid_ratio = 0.1, test_ratio=0.1)
    #     print("Test set samples: ", len(test_data))
    #     test_loader = DataLoader(dataset=test_data, batch_size=opt.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    #     count = 0
    #     for i, (_, target, input) in enumerate(test_loader):
    #         input = input.cuda()
    #         target = target.cuda()
    #         H, W = input.shape[-2], input.shape[-1]
    #         if modelname == 'SNCWGANDTN' and (H != H_ or W != W_):
    #             self.G = DTN(in_dim=3, 
    #                     out_dim=31,
    #                     img_size=[H, W], 
    #                     window_size=8, 
    #                     n_block=[2,2,2,2], 
    #                     bottleblock = 4).to(device)
    #             H_ = H
    #             W_ = W
    #             self.load_checkpoint()
    #         with torch.no_grad():
    #             output = self.G(input)
    #             rgbs = []
    #             reals = []
    #             for j in range(output.shape[0]):
    #                 mat = {}
    #                 mat['cube'] = np.transpose(target[j,:,:,:].cpu().numpy(), [1,2,0])
    #                 mat['rgb'] = np.transpose(input[j,:,:,:].cpu().numpy(), [1,2,0])
    #                 real = mat['rgb']
    #                 real = (real - real.min()) / (real.max()-real.min())
    #                 mat['rgb'] = real
    #                 rgb = SaveSpectral(output[j,:,:,:], count, root='/work3/s212645/Spectral_Reconstruction/FakeHyperSpectrum/' + modelname + '/FullResol/')
    #                 rgbs.append(rgb)
    #                 reals.append(real)
    #                 count += 1
    #             loss_mrae = criterion_mrae(output, target)
    #             loss_rmse = criterion_rmse(output, target)
    #             loss_psnr = criterion_psnr(output, target)
    #             loss_sam = criterion_sam(output, target)
    #             loss_sid = criterion_sid(output, target)
    #             rgbs = np.array(rgbs).transpose(0, 3, 1, 2)
    #             rgbs = torch.from_numpy(rgbs).cuda()
    #             reals = np.array(reals).transpose(0, 3, 1, 2)
    #             reals = torch.from_numpy(reals).cuda()
    #             # loss_fid = criterion_fid(rgbs, reals)
    #             loss_ssim = criterion_ssim(rgbs, reals)
    #             loss_psrnrgb = criterion_psnrrgb(rgbs, reals)
    #         # record loss
    #         losses_mrae.update(loss_mrae.data)
    #         losses_rmse.update(loss_rmse.data)
    #         losses_psnr.update(loss_psnr.data)
    #         losses_sam.update(loss_sam.data)
    #         losses_sid.update(loss_sid.data)
    #         # losses_fid.update(loss_fid.data)
    #         losses_ssim.update(loss_ssim.data)
    #         losses_psnrrgb.update(loss_psrnrgb.data)
    #     criterion_sam.reset()
    #     criterion_psnr.reset()
    #     criterion_psnrrgb.reset()
    #     file = '/zhome/02/b/164706/Master_Courses/2023_Fall/Spectral_Reconstruction/result.txt'
    #     f = open(file, 'a')
    #     f.write(modelname+':\n')
    #     f.write(f'MRAE:{losses_mrae.avg}, RMSE: {losses_rmse.avg}, PNSR:{losses_psnr.avg}, SAM: {losses_sam.avg}, SID: {losses_sid.avg}, SSIM: {losses_ssim.avg}, PSNRRGB: {losses_psnrrgb.avg}')
    #     f.write('\n')
    #     return losses_mrae.avg, losses_rmse.avg, losses_psnr.avg, losses_sam.avg, losses_sid.avg, losses_fid.avg, losses_ssim.avg, losses_psnrrgb.avg


if __name__ == '__main__':
    # spec = SpectralNormalizationCGAN()
    data = instantiate_from_config(cfg.data)
    model = instantiate_from_config(cfg.model)
    trainer = Trainer(accelerator="gpu", strategy="ddp_find_unused_parameters_true", default_root_dir="/work3/s212645/Spectral_Reconstruction/checkpoint/SNCWGAN/DTN_/")
    # trainer = Trainer(devices=[1], accelerator="gpu", strategy="ddp")
    trainer.fit(model, data)
    # if opt.loadmodel:
    #     try:
    #         spec.load_checkpoint()
    #     except:
    #         print('pretrained model loading failed')
    # if opt.mode == 'train':
    #     spec.train()
    #     spec.load_checkpoint(best=True)
    #     spec.test('SNCWGAN'+spec.opt.G)
    #     spec.test_full_resol('SNCWGAN'+spec.opt.G)
    #     # spec.test('SNCWGAN'+spec.opt.G)
    # elif opt.mode == 'test':
    #     spec.load_checkpoint(best=True)
    #     spec.test('SNCWGAN'+spec.opt.G)
    # elif opt.mode == 'testfull':
    #     spec.load_checkpoint(best=True)
    #     spec.test_full_resol('SNCWGAN'+spec.opt.G)
