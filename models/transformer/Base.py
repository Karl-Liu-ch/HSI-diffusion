import sys
sys.path.append('./')
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.optim as optim
import torch.backends.cudnn as cudnn
from dataset.datasets import TrainDataset, ValidDataset, TestDataset
from utils import *
import os
from torch.utils.data import DataLoader
from torch.autograd import Variable
import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
from utils import instantiate_from_config
from omegaconf import OmegaConf
from options import opt
import numpy as np
from color_loss import deltaELoss
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
if opt.multigpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# loss function
criterion_mrae = Loss_MRAE()
criterion_rmse = Loss_RMSE()
criterion_psnr = Loss_PSNR()
criterion_psnrrgb = Loss_PSNR()
criterion_sam = Loss_SAM()
criterion_sid = Loss_SID()
criterion_fid = Loss_Fid().to(device)
criterion_ssim = Loss_SSIM().to(device)


class BaseModel(pl.LightningModule):
    def __init__(self,
                 epochs,
                 learning_rate,
                 cond_key, 
                 modelconfig,
                 ):
        super().__init__()
        self.opt = opt
        self.epoch = 0
        self.end_epoch = epochs
        self.iteration = 0
        self.model = instantiate_from_config(modelconfig)
        self.best_mrae = 1000
        self.name = modelconfig.target.split('.')[-1]
        self.criterion = criterion_mrae
        self.root = '/work3/s212645/Spectral_Reconstruction/checkpoint/'+self.name+'/'
        self.learning_rate = learning_rate
        self.cond_key = cond_key
        # make checkpoint dir
        if not os.path.exists(opt.outf):
            os.makedirs(opt.outf)
        if not os.path.exists(self.root):
            os.makedirs(self.root)
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

    def forward(self, input):
        reconstructions = self.model(input)
        return reconstructions

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

    def modify_delta_e_gradients(self, images, labels):
        optimizer = self.optimizers()
        optimizer.zero_grad()
        outputs = self.model(images)
        loss1 = deltaELoss(outputs, labels)
        loss1.backward()

        gradients1 = [param.grad.clone() for param in self.model.parameters()]

        optimizer.zero_grad()
        outputs = self.model(images)
        loss2 = self.criterion(outputs, labels)
        loss2.backward()

        mask_gradients = []
        for grad1, grad2 in zip(gradients1, self.model.parameters()):
            mask1 = torch.abs(torch.sign(grad1) + torch.sign(grad2.grad)) / 2.0
            mask2 = - torch.abs(torch.sign(grad1) - torch.sign(grad2.grad)) / 2.0
            mask = mask1 + mask2
            mask_gradients.append(grad2.grad + mask * grad1)

        for param, grad in zip(self.model.parameters(), mask_gradients):
            param.grad = grad

        optimizer.step()

        return loss2, loss1

    def training_step(self, batch, batch_idx):
        images = batch[self.cond_key]
        labels = batch['label']
        images = Variable(images)
        labels = Variable(labels)
        # loss_mrae, loss = self.modify_delta_e_gradients(images, labels)
        # self.optimizer.zero_grad()
        output = self.model(images)
        loss_mrae = criterion_mrae(output, labels)
        self.log("train/loss", loss_mrae.detach(), prog_bar=True, logger=True, on_step=True, on_epoch=True)
        log_dict_g = {'mrae': loss_mrae}
        self.log_dict(log_dict_g, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return loss_mrae
            
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
        output = self.model(images)
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
        if batch_idx == 0:
            print(f'validation: MRAE: {self.losses_mrae.avg}, RMSE: {self.losses_rmse.avg}, PSNR: {self.losses_psnr.avg}, SAM: {self.losses_sam.avg}.')
        return self.log_dict
    
    def configure_optimizers(self):
        lr = self.learning_rate
        opt = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.5, 0.9))
        return [opt], []

    # def test(self):
    #     modelname = self.name
    #     try: 
    #         os.mkdir('/work3/s212645/Spectral_Reconstruction/FakeHyperSpectrum/')
    #         os.mkdir('/work3/s212645/Spectral_Reconstruction/RealHyperSpectrum/')
    #     except:
    #         pass
    #     root = '/work3/s212645/Spectral_Reconstruction/RealHyperSpectrum/' + modelname + '/'
    #     try: 
    #         os.mkdir('/work3/s212645/Spectral_Reconstruction/FakeHyperSpectrum/' + modelname + '/')
    #         os.mkdir('/work3/s212645/Spectral_Reconstruction/RealHyperSpectrum/' + modelname + '/')
    #     except:
    #         pass
    #     test_data = TestDataset(data_root=opt.data_root, crop_size=opt.patch_size, valid_ratio = 0.1, test_ratio=0.1)
    #     print("Test set samples: ", len(test_data))
    #     test_loader = DataLoader(dataset=test_data, batch_size=opt.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    #     self.model.eval()
    #     losses_mrae = AverageMeter()
    #     losses_rmse = AverageMeter()
    #     losses_psnr = AverageMeter()
    #     losses_psnrrgb = AverageMeter()
    #     losses_sam = AverageMeter()
    #     losses_sid = AverageMeter()
    #     losses_fid = AverageMeter()
    #     losses_ssim = AverageMeter()
    #     count = 0
    #     for i, (input, target) in enumerate(test_loader):
    #         input = input.cuda()
    #         target = target.cuda()
    #         with torch.no_grad():
    #             # compute output
    #             output = self.model(input)
    #             rgbs = []
    #             reals = []
    #             for j in range(output.shape[0]):
    #                 mat = {}
    #                 mat['cube'] = np.transpose(target[j,:,:,:].cpu().numpy(), [1,2,0])
    #                 mat['rgb'] = np.transpose(input[j,:,:,:].cpu().numpy(), [1,2,0])
    #                 real = mat['rgb']
    #                 real = (real - real.min()) / (real.max()-real.min())
    #                 mat['rgb'] = real
    #                 rgb = SaveSpectral(output[j,:,:,:], count, root='/work3/s212645/Spectral_Reconstruction/FakeHyperSpectrum/' + modelname + '/')
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
    #             loss_fid = criterion_fid(rgbs, reals)
    #             loss_ssim = criterion_ssim(rgbs, reals)
    #             loss_psrnrgb = criterion_psnrrgb(rgbs, reals)
    #         # record loss
    #         losses_mrae.update(loss_mrae.data)
    #         losses_rmse.update(loss_rmse.data)
    #         losses_psnr.update(loss_psnr.data)
    #         losses_sam.update(loss_sam.data)
    #         losses_sid.update(loss_sid.data)
    #         losses_fid.update(loss_fid.data)
    #         losses_ssim.update(loss_ssim.data)
    #         losses_psnrrgb.update(loss_psrnrgb.data)
    #     criterion_sam.reset()
    #     criterion_psnr.reset()
    #     criterion_psnrrgb.reset()
    #     file = '/zhome/02/b/164706/Master_Courses/2023_Fall/Spectral_Reconstruction/result.txt'
    #     f = open(file, 'a')
    #     f.write(modelname+':\n')
    #     f.write(f'MRAE:{losses_mrae.avg}, RMSE: {losses_rmse.avg}, PNSR:{losses_psnr.avg}, SAM: {losses_sam.avg}, SID: {losses_sid.avg}, FID: {losses_fid.avg}, SSIM: {losses_ssim.avg}, PSNRRGB: {losses_psnrrgb.avg}')
    #     f.write('\n')
    #     return losses_mrae.avg, losses_rmse.avg, losses_psnr.avg, losses_sam.avg, losses_sid.avg, losses_fid.avg, losses_ssim.avg, losses_psnrrgb.avg
    
    # def save_metrics(self):
    #     name = 'metrics.pth'
    #     torch.save(self.metrics, self.root+name)
        
    # def load_metrics(self):
    #     name = 'metrics.pth'
    #     checkpoint = torch.load(os.path.join(self.root, name))
    #     self.metrics = checkpoint
    
    # def save_checkpoint(self, best = False):
    #     if self.multiGPU:
    #         state = {
    #             'epoch': self.epoch,
    #             'iter': self.iteration,
    #             'best_mrae': self.best_mrae, 
    #             'state_dict': self.model.module.state_dict(),
    #             'optimizer': self.optimizer.state_dict(),
    #         }
    #     else:
    #         state = {
    #             'epoch': self.epoch,
    #             'iter': self.iteration,
    #             'best_mrae': self.best_mrae, 
    #             'state_dict': self.model.state_dict(),
    #             'optimizer': self.optimizer.state_dict(),
    #         }

    #     if best: 
    #         name = 'net_epoch_best.pth'
    #         torch.save(state, os.path.join(self.root, name))
    #     name = 'net.pth'
    #     torch.save(state, os.path.join(self.root, name))
        
    # def load_checkpoint(self, best = False):
    #     if best:
    #         checkpoint = torch.load(os.path.join(self.root, 'net_epoch_best.pth'))
    #     else:
    #         checkpoint = torch.load(os.path.join(self.root, 'net.pth'))
    #     if self.multiGPU:
    #         self.model.module.load_state_dict(checkpoint['state_dict'])
    #     else:
    #         self.model.load_state_dict(checkpoint['state_dict'])
    #     self.optimizer.load_state_dict(checkpoint['optimizer'])
    #     self.iteration = checkpoint['iter']
    #     self.epoch = checkpoint['epoch']
    #     self.best_mrae = checkpoint['best_mrae']
    #     try:
    #         self.load_metrics()
    #     except:
    #         pass
    #     print("pretrained model loaded")

class CVAECPModel(BaseModel):
    def train(self):
        self.load_dataset()
        self.model.train()
        record_mrae_loss = 1000
        while self.epoch<self.end_epoch:
            losses = AverageMeter()
            train_loader = DataLoader(dataset=self.train_data, batch_size=self.opt.batch_size, shuffle=True, num_workers=2,
                                    pin_memory=True, drop_last=True)
            val_loader = DataLoader(dataset=self.val_data, batch_size=self.opt.batch_size, shuffle=False, num_workers=2, pin_memory=True)
            for i, (images, labels) in enumerate(train_loader):
                labels = labels.to(device)
                images = images.to(device)
                images = Variable(images)
                labels = Variable(labels)
                lr = self.optimizer.param_groups[0]['lr']
                self.optimizer.zero_grad()
                [output, input, mu0, log_var0, mu1, log_var1, mu2, log_var2] = self.model(labels, images)
                if self.multiGPU:
                    Loss_dict = self.model.module.loss_function(output, input, mu0, log_var0, mu1, log_var1, mu2, log_var2)
                else:
                    Loss_dict = self.model.loss_function(output, input, mu0, log_var0, mu1, log_var1, mu2, log_var2)
                loss = Loss_dict['loss']
                loss_mrae = criterion_mrae(output, input)
                loss += loss_mrae
                loss.backward()
                losses.update(loss_mrae.data)
                self.optimizer.step()
                self.iteration = self.iteration+1
                if self.iteration % 20 == 0:
                    print('[epoch:%d/%d],lr=%.9f,train_losses.avg=%.9f'
                        % (self.epoch, self.end_epoch, lr, losses.avg))
            # if self.iteration % 1000 == 0:
            mrae_loss, rmse_loss, psnr_loss, sam_loss, sid_loss = self.validate(val_loader)
            print(f'MRAE:{mrae_loss}, RMSE: {rmse_loss}, PNSR:{psnr_loss}, SAM: {sam_loss}, SID: {sid_loss}')
            # Save model
            # if torch.abs(mrae_loss - record_mrae_loss) < 0.01 or mrae_loss < record_mrae_loss or self.iteration % 5000 == 0:
            print(f'Saving to {self.root}')
            self.save_checkpoint()
            if mrae_loss < record_mrae_loss:
                record_mrae_loss = mrae_loss
                self.save_checkpoint(best=True)
            # print loss
            print(" Iter[%06d], Epoch[%06d], learning rate : %.9f, Train MRAE: %.9f, Test MRAE: %.9f, "
                "Test RMSE: %.9f, Test PSNR: %.9f, SAM: %.9f, SID: %.9f " % (self.iteration, 
                                                                self.epoch, lr, 
                                                                losses.avg, mrae_loss, rmse_loss, psnr_loss, sam_loss, sid_loss))
            self.epoch += 1
            self.scheduler.step()

if __name__ == '__main__':
    cfg = OmegaConf.load("configs/mst.yaml")
    data = instantiate_from_config(cfg.data)
    model = instantiate_from_config(cfg.model)
    trainer = Trainer(accelerator="gpu", strategy="ddp_find_unused_parameters_true", default_root_dir="/work3/s212645/Spectral_Reconstruction/checkpoint/MST/")
    trainer.fit(model, data)