import sys
sys.path.append('./')
import torch.nn as nn
import torch
torch.manual_seed(1234)
from options import opt
import os
from torch.utils.data import DataLoader
from torch.autograd import Variable
from models.gan.networks import *
from models.transformer.DTN import DTN
import numpy as np
from dataset.datasets import TestFullDataset
from models.gan.Basemodel import BaseModel, criterion_mrae, AverageMeter, SAM
from utils import AverageMeter, record_loss, Loss_MRAE, Loss_RMSE, Loss_PSNR, Loss_Fid, \
    Loss_SAM, Loss_SSIM, reconRGB, Loss_SID, SAM, SaveSpectral
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
if opt.multigpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NONOISE = opt.nonoise
criterion_mrae = Loss_MRAE()
criterion_rmse = Loss_RMSE()
criterion_psnr = Loss_PSNR()
criterion_psnrrgb = Loss_PSNR()
criterion_sam = Loss_SAM()
criterion_sid = Loss_SID()
criterion_fid = Loss_Fid().cuda()
criterion_ssim = Loss_SSIM().cuda()
torch.autograd.set_detect_anomaly(True)

class SwinGAN(BaseModel):
    def __init__(self, opt, multiGPU=False):
        super().__init__(opt, multiGPU)
        self.lossl1 = nn.L1Loss()
        self.lambdal1 = 40
        self.lambdasam = 1
        self.lambdaperceptual = 0.0
        self.lambda_disc = 0.0
        self.root = f'/work3/s212645/Spectral_Reconstruction/checkpoint/SwinGAN/{self.opt.G}_{self.lambdal1}_{self.lambdasam}_{self.lambdaperceptual}_{self.lambda_disc}/'
        self.nonoise = True
        self.init_Net()
        self.init_metrics()
    
    def init_Net(self):
        self.G = DTN(in_dim=6, 
                out_dim=31,
                img_size=[opt.patch_size, opt.patch_size], 
                window_size=8, 
                n_block=[2,2,2,2], 
                bottleblock = 4)
        # self.G = DTN(in_dim=3, 
        #         out_dim=31,
        #         img_size=[opt.patch_size, opt.patch_size], 
        #         window_size=8, 
        #         n_block=[2,2,2,2], 
        #         bottleblock = 4)
        self.nonoise = True
        print('DTN, No noise')
        # self.D = SN_Discriminator(34)
        # self.D = SN_Discriminator_perceptualLoss(37)
        # self.D = SNResnetDiscriminator_perceptualLoss(37)
        self.D = SNTransformDiscriminator_perceptual(37)
        super().init_Net()
      
    def get_last_layer(self):
        return self.G.mapping.weight
      
    def calculate_adaptive_weight(self, rec_loss, g_loss, last_layer=None):
        if last_layer is not None:
            rec_grads = torch.autograd.grad(rec_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            rec_grads = torch.autograd.grad(rec_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(rec_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        # d_weight = d_weight * self.discriminator_weight
        return d_weight

    def train(self):
        super().train()
        self.load_dataset(datanames=['ARAD/'])
        while self.epoch<self.end_epoch:
            self.G.train()
            self.D.train()
            losses = AverageMeter()
            train_loader = DataLoader(dataset=self.train_data, batch_size=self.opt.batch_size, shuffle=True, num_workers=32,
                                    pin_memory=True, drop_last=True)
            val_loader = DataLoader(dataset=self.val_data, batch_size=self.opt.batch_size, shuffle=False, num_workers=32, pin_memory=True)
            for i, batch in enumerate(train_loader):
                labels = batch['label'].cuda()
                images = batch['ycrcb'].cuda()
                images = Variable(images)
                labels = Variable(labels)
                if self.nonoise:
                    z = images
                else:
                    z = torch.randn_like(images).cuda()
                    z = torch.concat([z, images], dim=1)
                    z = Variable(z)
                realAB = torch.concat([images, labels], dim=1)
                # D_real, D_real_feature = self.D(realAB)
                x_fake = self.G(z)
                fakeAB = torch.concat([images, x_fake],dim=1)
                
                # train D
                for p in self.D.parameters():
                    p.requires_grad = True
                self.optimD.zero_grad()
                D_real, D_real_feature = self.D(realAB)
                loss_real = -D_real.mean(0).view(1)
                D_fake, D_fake_feature = self.D(fakeAB.detach())
                loss_fake = D_fake.mean(0).view(1)
                perceptual_loss = 0
                for k in range(len(D_fake_feature)):
                    perceptual_loss -= nn.MSELoss()(D_real_feature[k], D_fake_feature[k])
                loss_d = loss_real + loss_fake + perceptual_loss * self.lambda_disc
                loss_d.backward()
                self.optimD.step()
                
                # train G
                self.optimG.zero_grad()
                lrG = self.optimG.param_groups[0]['lr']
                for p in self.D.parameters():
                    p.requires_grad = False
                pred_fake, D_fake_feature = self.D(fakeAB)
                loss_G = -pred_fake.mean(0).view(1)
                lossl1 = self.lossl1(x_fake, labels) * self.lamda
                losssam = SAM(x_fake, labels) * self.lambdasam
                perceptual_loss = 0
                for k in range(len(D_fake_feature)):
                    perceptual_loss += nn.MSELoss()(D_real_feature[k].detach(), D_fake_feature[k])
                weight_gan = self.calculate_adaptive_weight(lossl1 + losssam, loss_G, self.get_last_layer())
                loss_G *= weight_gan
                loss_G += lossl1 + losssam + perceptual_loss * self.lambdaperceptual * weight_gan / len(D_fake_feature)
                # train the generator
                loss_G.backward()
                self.optimG.step()
                
                loss_mrae = criterion_mrae(x_fake, labels)
                losses.update(loss_mrae.data)
                self.iteration = self.iteration+1
                if self.iteration % 20 == 0:
                    print('[epoch:%d/%d],lr=%.9f,train_losses.avg=%.9f,disc losses=%.9f'
                        % (self.epoch, self.end_epoch, lrG, losses.avg, -pred_fake.mean(0).view(1)))
            # validation
            mrae_loss, rmse_loss, psnr_loss, sam_loss, sid_loss = self.validate(val_loader)
            print(f'MRAE:{mrae_loss}, RMSE: {rmse_loss}, PNSR:{psnr_loss}, SAM: {sam_loss}, SID: {sid_loss}')
            # Save model
            print(f'Saving to {self.root}')
            self.save_checkpoint()
            if mrae_loss < self.best_mrae:
                self.best_mrae = mrae_loss
                self.save_checkpoint(True)
            # save_top, top_n = self.save_top_k(mrae_loss.detach().cpu().numpy())
            # if save_top:
            #     self.save_checkpoint(top_k=top_n)
            # print loss
            print(" Iter[%06d], Epoch[%06d], learning rate : %.9f, Train MRAE: %.9f, Test MRAE: %.9f, "
                "Test RMSE: %.9f, Test PSNR: %.9f, SAM: %.9f, SID: %.9f " % (self.iteration, 
                                                                self.epoch, lrG, 
                                                                losses.avg, mrae_loss, rmse_loss, psnr_loss, sam_loss, sid_loss))
            self.epoch += 1
            self.schedulerD.step()
            self.schedulerG.step()
            if mrae_loss > 100.0: 
                break

    def test_full_resol(self, modelname):
        try:
            os.mkdir('/work3/s212645/Spectral_Reconstruction/FakeHyperSpectrum/' + modelname + '/FullResol/')
        except:
            pass
        root = '/work3/s212645/Spectral_Reconstruction/RealHyperSpectrum/'
        H_ = 128
        W_ = 128
        losses_mrae = AverageMeter()
        losses_rmse = AverageMeter()
        losses_psnr = AverageMeter()
        losses_psnrrgb = AverageMeter()
        losses_sam = AverageMeter()
        losses_sid = AverageMeter()
        losses_fid = AverageMeter()
        losses_ssim = AverageMeter()
        test_data = TestFullDataset(data_root=opt.data_root, crop_size=opt.patch_size, valid_ratio = 0.1, test_ratio=0.1)
        print("Test set samples: ", len(test_data))
        test_loader = DataLoader(dataset=test_data, batch_size=opt.batch_size, shuffle=False, num_workers=32, pin_memory=True)
        count = 0
        for i, batch in enumerate(test_loader):
            target = batch['label'].cuda()
            input = batch['ycrcb'].cuda()
            rgb_gt = batch['cond'].cuda()
            H, W = input.shape[-2], input.shape[-1]
            if modelname == 'SNCWGANDTN' and (H != H_ or W != W_):
                self.G = DTN(in_dim=6, 
                        out_dim=31,
                        img_size=[H, W], 
                        window_size=8, 
                        n_block=[2,2,2,2], 
                        bottleblock = 4).to(device)
                H_ = H
                W_ = W
                self.load_checkpoint(best=True)
            with torch.no_grad():
                output = self.G(input)
                rgbs = []
                reals = []
                for j in range(output.shape[0]):
                    mat = {}
                    mat['cube'] = np.transpose(target[j,:,:,:].cpu().numpy(), [1,2,0])
                    mat['rgb'] = np.transpose(input[j,:,:,:].cpu().numpy(), [1,2,0])
                    real = np.transpose(rgb_gt[j,:,:,:].cpu().numpy(), [1,2,0])
                    real = (real - real.min()) / (real.max()-real.min())
                    mat['rgb'] = real
                    rgb = SaveSpectral(output[j,:,:,:], count, root='/work3/s212645/Spectral_Reconstruction/FakeHyperSpectrum/' + modelname + '/FullResol/')
                    rgbs.append(rgb)
                    reals.append(real)
                    count += 1
                loss_mrae = criterion_mrae(output, target)
                loss_rmse = criterion_rmse(output, target)
                loss_psnr = criterion_psnr(output, target)
                loss_sam = criterion_sam(output, target)
                loss_sid = criterion_sid(output, target)
                rgbs = np.array(rgbs).transpose(0, 3, 1, 2)
                rgbs = torch.from_numpy(rgbs).cuda()
                reals = np.array(reals).transpose(0, 3, 1, 2)
                reals = torch.from_numpy(reals).cuda()
                # loss_fid = criterion_fid(rgbs, reals)
                loss_ssim = criterion_ssim(rgbs, reals)
                loss_psrnrgb = criterion_psnrrgb(rgbs, reals)
            # record loss
            losses_mrae.update(loss_mrae.data)
            losses_rmse.update(loss_rmse.data)
            losses_psnr.update(loss_psnr.data)
            losses_sam.update(loss_sam.data)
            losses_sid.update(loss_sid.data)
            # losses_fid.update(loss_fid.data)
            losses_ssim.update(loss_ssim.data)
            losses_psnrrgb.update(loss_psrnrgb.data)
        criterion_sam.reset()
        criterion_psnr.reset()
        criterion_psnrrgb.reset()
        file = '/zhome/02/b/164706/Master_Courses/2023_Fall/Spectral_Reconstruction/result.txt'
        f = open(file, 'a')
        f.write(modelname+':\n')
        f.write(f'MRAE:{losses_mrae.avg}, RMSE: {losses_rmse.avg}, PNSR:{losses_psnr.avg}, SAM: {losses_sam.avg}, SID: {losses_sid.avg}, SSIM: {losses_ssim.avg}, PSNRRGB: {losses_psnrrgb.avg}')
        f.write('\n')
        return losses_mrae.avg, losses_rmse.avg, losses_psnr.avg, losses_sam.avg, losses_sid.avg, losses_fid.avg, losses_ssim.avg, losses_psnrrgb.avg


if __name__ == '__main__':
    spec = SwinGAN(opt, multiGPU=opt.multigpu)
    # if opt.loadmodel:
    #     try:
    #         spec.load_checkpoint(best=False)
    #     except:
    #         print('pretrained model loading failed')
    if opt.mode == 'train':
        spec.train()
        spec.load_checkpoint(best=True)
        spec.test('SNCWGAN'+spec.opt.G, datanames=['ARAD/'])
        spec.test_full_resol('SNCWGAN'+spec.opt.G)
        # spec.test('SNCWGAN'+spec.opt.G)
    elif opt.mode == 'test':
        spec.load_checkpoint(best=True)
        spec.test('SNCWGAN'+spec.opt.G, datanames=['ARAD/'])
    elif opt.mode == 'testfull':
        spec.load_checkpoint(best=True)
        spec.test_full_resol('SNCWGAN'+spec.opt.G)
