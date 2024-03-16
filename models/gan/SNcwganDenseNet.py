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
from utils import AverageMeter, record_loss, Loss_MRAE, Loss_RMSE, Loss_PSNR, Loss_Fid, \
    Loss_SAM, Loss_SSIM, reconRGB, Loss_SID, SAMLoss, SAM
from options import opt
import os
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import autograd
import functools
from Models.GAN.networks import *
import numpy as np
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
# criterion_fid = Loss_Fid().cuda()
# criterion_ssim = Loss_SSIM().cuda()

class SNCWGANDenseNet():
    def __init__(self, opt, multiGPU = False) -> None:
        super().__init__()
        self.multiGPU = multiGPU
        self.opt = opt
        self.G = DensenetGenerator(inchannels = 6, 
                 outchannels = 31, 
                 num_init_features = 64, 
                 block_config = (6, 12, 24, 16),
                 bn_size = 4, 
                 growth_rate = 32, 
                 center_layer = 6
                 )
        self.D = SN_Discriminator(34)
        if self.multiGPU:
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)
        self.G.cuda()
        self.D.cuda()
        
        self.epoch = 0
        self.end_epoch = opt.end_epoch
        per_epoch_iteration = 1000
        self.total_iteration = per_epoch_iteration*opt.end_epoch
        self.iteration = 0
        
        self.optimG = optim.Adam(self.G.parameters(), lr=self.opt.init_lr, betas=(0.9, 0.999))
        self.schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimG, self.end_epoch, eta_min=1e-6)
        self.optimD = optim.Adam(self.D.parameters(), lr=self.opt.init_lr, betas=(0.9, 0.999))
        self.schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimD, self.end_epoch, eta_min=1e-6)
        self.lossl1 = nn.L1Loss()
        self.lamda = 100
        self.lambdasam = 0
        self.root = '/work3/s212645/Spectral_Reconstruction/checkpoint/SNCWGANDenseNet/'
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        self.metrics = {
            'MRAE':np.zeros(shape=[self.end_epoch]),
            'RMSE':np.zeros(shape=[self.end_epoch]),
            'PSNR':np.zeros(shape=[self.end_epoch]),
            'SAM':np.zeros(shape=[self.end_epoch])
        }
    
    def load_dataset(self):
        # load dataset
        print("\nloading dataset ...")
        self.train_data = TrainDataset(data_root=self.opt.data_root, crop_size=self.opt.patch_size, valid_ratio = 0.1, test_ratio=0.1)
        print(f"Iteration per epoch: {len(self.train_data)}")
        self.val_data = ValidDataset(data_root=self.opt.data_root, crop_size=self.opt.patch_size, valid_ratio = 0.1, test_ratio=0.1)
        print("Validation set samples: ", len(self.val_data))
        self.test_data = TestDataset(data_root=self.opt.data_root, crop_size=self.opt.patch_size, valid_ratio = 0.1, test_ratio=0.1)
        print("Validation set samples: ", len(self.val_data))
    
    def calculate_gradient_penalty(self, real_labels, fake_labels, images):
        eta = torch.FloatTensor(real_labels.size(0),1,1,1).uniform_(0,1).cuda()
        eta = eta.expand(real_labels.size(0), real_labels.size(1), real_labels.size(2), real_labels.size(3))

        interpolated = eta * real_labels + ((1 - eta) * fake_labels)

        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        interpolated = torch.concat([images, interpolated], dim=1)
        prob_interpolated = self.D(interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(
                                   prob_interpolated.size()).cuda(),
                               create_graph=True, retain_graph=True)[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_term
        return grad_penalty
    
    def train(self):
        self.load_dataset()
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
                z = torch.randn_like(images).cuda()
                z = torch.concat([z, images], dim=1)
                z = Variable(z)
                realAB = torch.concat([images, labels], dim=1)
                D_real = self.D(realAB)
                real_labels = torch.ones_like(D_real).cuda()
                x_fake = self.G(z)
                fakeAB = torch.concat([images, x_fake],dim=1)
                fake_labels = torch.zeros_like(D_real).cuda()
                
                # train D
                for p in self.D.parameters():
                    p.requires_grad = True
                self.optimD.zero_grad()
                D_real = self.D(realAB)
                loss_real = -D_real.mean(0).view(1)
                loss_real.backward()
                D_fake = self.D(fakeAB.detach())
                loss_fake = D_fake.mean(0).view(1)
                loss_fake.backward()
                # gp = self.calculate_gradient_penalty(labels.data, x_fake.data, images.data)
                # gp.backward()
                self.optimD.step()
                
                # train G
                self.optimG.zero_grad()
                lrG = self.optimG.param_groups[0]['lr']
                for p in self.D.parameters():
                    p.requires_grad = False
                pred_fake = self.D(fakeAB)
                loss_G = -pred_fake.mean(0).view(1)
                lossl1 = self.lossl1(x_fake, labels) * self.lamda
                losssam = SAM(x_fake, labels) * self.lambdasam
                loss_G += lossl1 + losssam
                # train the generator
                loss_G.backward()
                self.optimG.step()
                
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
                                                                self.epoch, lrG, 
                                                                losses.avg, mrae_loss, rmse_loss, psnr_loss, sam_loss, sid_loss))
            self.epoch += 1
            self.schedulerD.step()
            self.schedulerG.step()
                
    def validate(self, val_loader):
        self.G.eval()
        losses_mrae = AverageMeter()
        losses_rmse = AverageMeter()
        losses_psnr = AverageMeter()
        losses_sam = AverageMeter()
        losses_sid = AverageMeter()
        # losses_fid = AverageMeter()
        # losses_ssim = AverageMeter()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()
            z = torch.randn_like(input).cuda()
            z = torch.concat([z, input], dim=1)
            z = Variable(z)
            with torch.no_grad():
                # compute output
                output = self.G(z)
                loss_mrae = criterion_mrae(output, target)
                loss_rmse = criterion_rmse(output, target)
                loss_psnr = criterion_psnr(output, target)
                loss_sam = criterion_sam(output, target)
                loss_sid = criterion_sid(output, target)
                # rgb = reconRGB(output)
                # loss_fid = criterion_fid(rgb, input)
                # loss_ssim = criterion_ssim(rgb, input)
            # record loss
            losses_mrae.update(loss_mrae.data)
            losses_rmse.update(loss_rmse.data)
            losses_psnr.update(loss_psnr.data)
            losses_sam.update(loss_sam.data)
            losses_sid.update(loss_sid.data)
            # losses_fid.update(loss_fid.data)
            # losses_ssim.update(loss_ssim.data)
        criterion_sam.reset()
        criterion_psnr.reset()
        self.metrics['MRAE'][self.epoch]=losses_mrae.avg.cpu().detach().numpy()
        self.metrics['RMSE'][self.epoch]=losses_rmse.avg.cpu().detach().numpy()
        self.metrics['PSNR'][self.epoch]=losses_psnr.avg.cpu().detach().numpy()
        self.metrics['SAM'][self.epoch]=losses_sam.avg.cpu().detach().numpy()
        self.save_metrics()
        return losses_mrae.avg, losses_rmse.avg, losses_psnr.avg, losses_sam.avg, losses_sid.avg
    
    def test(self, test_loader):
        pass
    
    def save_metrics(self):
        name = 'metrics.pth'
        torch.save(self.metrics, os.path.join(self.root, name))
        
    def load_metrics(self):
        name = 'metrics.pth'
        checkpoint = torch.load(os.path.join(self.root, name))
        self.metrics = checkpoint
    
    def save_checkpoint(self, best = False):
        if self.multiGPU:
            state = {
                'epoch': self.epoch,
                'iter': self.iteration,
                'G': self.G.module.state_dict(),
                'D': self.D.module.state_dict(),
                'optimG': self.optimG.state_dict(),
                'optimD': self.optimD.state_dict(),
            }
        else:
            state = {
                'epoch': self.epoch,
                'iter': self.iteration,
                'G': self.G.state_dict(),
                'D': self.D.state_dict(),
                'optimG': self.optimG.state_dict(),
                'optimD': self.optimD.state_dict(),
            }
        if best: 
            name = 'net_epoch_best.pth'
        else:
            name = 'net_%04depoch.pth' % self.epoch
            oldname = 'net_%04depoch.pth' % (self.epoch - 5)
            if os.path.exists(os.path.join(self.root, oldname)):
                os.remove(os.path.join(self.root, oldname))
                print(oldname, ' Removed. ')
        torch.save(state, os.path.join(self.root, name))
        
    def load_checkpoint(self):
        checkpoint = torch.load(os.path.join(self.root, 'net_epoch_best.pth'))
        if self.multiGPU:
            self.G.module.load_state_dict(checkpoint['G'])
            self.D.module.load_state_dict(checkpoint['D'])
        else:
            self.G.load_state_dict(checkpoint['G'])
            self.D.load_state_dict(checkpoint['D'])
        self.optimG.load_state_dict(checkpoint['optimG'])
        self.optimD.load_state_dict(checkpoint['optimD'])
        self.iteration = checkpoint['iter']
        self.epoch = checkpoint['epoch']
        try:
            self.load_metrics()
        except:
            pass
        print("pretrained model loaded, iteration: ", self.iteration)

if __name__ == '__main__':
    # input = torch.randn([1, 6, 128, 128]).cuda()
    # D = Discriminator(31).cuda()
    # G = Generator(6, 31).cuda()
    
    # output = G(input)
    # print(output.shape)
    
    spec = SNCWGANDenseNet(opt, multiGPU=opt.multigpu)
    if opt.loadmodel:
        try:
            spec.load_checkpoint()
        except:
            print('pretrained model loading failed')
    spec.train()