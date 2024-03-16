import sys
sys.path.append('./')
import torch.nn as nn
import torch
from torch.nn import functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.optim as optim
import torch.backends.cudnn as cudnn
# from hsi_dataset import TrainDataset, ValidDataset
from dataset.datasets import TrainDataset, ValidDataset, TestDataset
from utils import AverageMeter, record_loss, Loss_MRAE, Loss_RMSE, Loss_PSNR, Loss_Fid, Loss_SAM, Loss_SSIM, reconRGB, Loss_SID, SAM
from options import opt
import os
from torch.utils.data import DataLoader
from torch.autograd import Variable
import functools
from models.gan.Basemodel import BaseModel
from models.gan.networks import *
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
criterion_sam = Loss_SAM()
criterion_sid = Loss_SID()
    
class pix2pixlsgan(BaseModel):
    def __init__(self, opt, multiGPU=False, lossname = 'ls'):
        super().__init__(opt, multiGPU)
        self.lossname = lossname
        self.lossl1 = criterion_mrae
        self.lamda = 100
        self.lambdasam = 100
        self.root = '/work3/s212645/Spectral_Reconstruction/checkpoint/pix2pix/'
        self.nonoise = False
        self.init_Net()
        self.init_metrics()
        self.nonoise = True
        
    def init_Net(self):
        self.G = UnetGenerator(3, 31)
        self.D = NLayerDiscriminator(34)
        if self.multiGPU:
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)
        self.G.cuda()
        self.D.cuda()
        if self.lossname == 'ls':
            self.criterionGAN = nn.MSELoss()
        elif self.lossname == 'bce':
            self.criterionGAN = nn.BCEWithLogitsLoss()
        super().init_Net()

    def train(self):
        super().train()
        record_mrae_loss = 1000
        while self.epoch<self.end_epoch:
            self.G.train()
            self.D.train()
            losses = AverageMeter()
            train_loader = DataLoader(dataset=self.train_data, batch_size=self.opt.batch_size, shuffle=True, num_workers=2,
                                    pin_memory=True, drop_last=True)
            val_loader = DataLoader(dataset=self.val_data, batch_size=self.opt.batch_size, shuffle=False, num_workers=2, pin_memory=True)
            for i, (images, labels) in enumerate(train_loader):
                labels = labels.cuda()
                images = images.cuda()
                images = Variable(images)
                labels = Variable(labels)
                realAB = torch.concat([images, labels], dim=1)
                x_fake = self.G(images)
                fakeAB = torch.concat([images, x_fake],dim=1)
                
                # train D
                for p in self.D.parameters():
                    p.requires_grad = True
                self.D.zero_grad()
                lrD = self.optimD.param_groups[0]['lr']
                self.optimD.zero_grad()
                # realAB = torch.concat([images, labels], dim=1)
                D_real = self.D(realAB)
                real_labels = torch.ones_like(D_real).cuda()
                fake_labels = torch.zeros_like(D_real).cuda()
                loss_real = self.criterionGAN(D_real, real_labels)
                loss_real.backward()
                # x_fake = self.G(images).detach()
                # fakeAB = torch.concat([images, x_fake],dim=1)
                D_fake = self.D(fakeAB.detach())
                loss_fake = self.criterionGAN(D_fake, fake_labels)
                loss_fake.backward()
                self.optimD.step()
                self.schedulerD.step()
                
                # train G
                self.G.zero_grad()
                self.optimG.zero_grad()
                lrG = self.optimG.param_groups[0]['lr']
                for p in self.D.parameters():
                    p.requires_grad = False
                # x_fake = self.G(images)
                # fakeAB = torch.concat([images, x_fake],dim=1)
                pred_fake = self.D(fakeAB)
                loss_G = self.criterionGAN(pred_fake, real_labels)
                lossl1 = self.lossl1(x_fake, labels) * self.lamda
                loss_sam = SAM(x_fake, labels) * self.lamda
                loss_G += lossl1 + loss_sam
                # train the generator
                loss_G.backward()
                self.optimG.step()
                self.schedulerG.step()
                
                loss_mrae = criterion_mrae(x_fake, labels)
                losses.update(loss_mrae.data)
                self.iteration = self.iteration+1
                if self.iteration % 20 == 0:
                    print('[iter:%d/%d],lr=%.9f,train_losses.avg=%.9f'
                        % (self.iteration, self.total_iteration, lrG, losses.avg))
            # validation
            mrae_loss, rmse_loss, psnr_loss, sam_loss, sid_loss = self.validate(val_loader)
            print(f'MRAE:{mrae_loss}, RMSE: {rmse_loss}, PNSR:{psnr_loss}, SAM: {sam_loss}, SID: {sid_loss}')
            # Save model
            # if torch.abs(mrae_loss - record_mrae_loss) < 0.01 or mrae_loss < record_mrae_loss or self.iteration % 5000 == 0:
            print(f'Saving to {self.root}')
            self.save_checkpoint()
            if mrae_loss < record_mrae_loss:
                record_mrae_loss = mrae_loss
                self.save_checkpoint(True)
            # print loss
            print(" Iter[%06d], Epoch[%06d], learning rate : %.9f, Train MRAE: %.9f, Test MRAE: %.9f, "
                "Test RMSE: %.9f, Test PSNR: %.9f, SAM: %.9f, SID: %.9f " % (self.iteration, 
                                                                self.epoch, lrD, 
                                                                losses.avg, mrae_loss, rmse_loss, psnr_loss, sam_loss, sid_loss))
            self.epoch += 1

if __name__ == '__main__':
    spec = pix2pixlsgan(opt, multiGPU=opt.multigpu)
    if opt.loadmodel:
        try:
            spec.load_checkpoint()
        except:
            print('pretrained model loading failed')
    if opt.mode == 'train':
        spec.train()
        spec.test('pix2pix')
    elif opt.mode == 'test':
        spec.test('pix2pix')