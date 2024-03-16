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
from utils import AverageMeter, record_loss, Loss_MRAE, Loss_RMSE, Loss_PSNR, Loss_Fid, Loss_SAM, Loss_SSIM, reconRGB, Loss_SID
from options import opt
import os
from torch.utils.data import DataLoader
from torch.autograd import Variable
import functools
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

class Encoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 condition_channels: int) -> None:
        super(Encoder, self).__init__()

        self.in_channels = in_channels

        def ConvT(input_nums, output_nums):
            layer = []
            layer.append(nn.ConvTranspose2d(input_nums, output_nums, kernel_size=(4,4), stride=(2,2), padding=(1,1), bias=False))
            layer.append(nn.BatchNorm2d(output_nums))
            layer.append(nn.ReLU())
            return layer
        
        def Conv(input_nums, output_nums, stride):
            layer = []
            layer.append(nn.Conv2d(input_nums, output_nums, kernel_size=(3,3), stride=stride, padding=(1,1), bias=False))
            layer.append(nn.BatchNorm2d(output_nums))
            layer.append(nn.ReLU())
            return layer
        
        hidden_dims = [64, 128]
        self.embed_class0 = nn.Conv2d(condition_channels, condition_channels, kernel_size=1, bias=False)
        self.embed_class1 = nn.Sequential(*Conv(condition_channels, hidden_dims[0], stride=2))
        self.embed_class2 = nn.Sequential(*Conv(hidden_dims[0], hidden_dims[1], stride=2))
        
        self.embed_data = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
            
        self.hidden_dims = hidden_dims
        # Build Encoder
        self.encconv1 = nn.Sequential(*Conv(in_channels, hidden_dims[0], stride=2))
        self.encconv2 = nn.Sequential(*Conv(hidden_dims[0], hidden_dims[1], stride=2))
        
        self.fc_mu0 = nn.Sequential(*Conv(in_channels, in_channels, stride=1), 
                                    nn.Conv2d(in_channels, in_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
                                    )
        self.fc_var0 = nn.Sequential(*Conv(in_channels, in_channels, stride=1), 
                                    nn.Conv2d(in_channels, in_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
                                    )
        self.fc_mu1 = nn.Sequential(*Conv(hidden_dims[0], hidden_dims[0], stride=1), 
                                    nn.Conv2d(hidden_dims[0], hidden_dims[0], kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
                                    )
        self.fc_var1 = nn.Sequential(*Conv(hidden_dims[0], hidden_dims[0], stride=1), 
                                    nn.Conv2d(hidden_dims[0], hidden_dims[0], kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
                                    )
        self.fc_mu2 = nn.Sequential(*Conv(hidden_dims[1], hidden_dims[1], stride=1), 
                                    nn.Conv2d(hidden_dims[1], hidden_dims[1], kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
                                    )
        self.fc_var2 = nn.Sequential(*Conv(hidden_dims[1], hidden_dims[1], stride=1), 
                                    nn.Conv2d(hidden_dims[1], hidden_dims[1], kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
                                    )
        
    def forward(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        embed_input = self.embed_data(input)
        
        embed_input1 = self.encconv1(embed_input)
        embed_input2 = self.encconv2(embed_input1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu0 = self.fc_mu0(embed_input)
        log_var0 = self.fc_var0(embed_input) # [31, H, W]
        mu1 = self.fc_mu1(embed_input1)
        log_var1 = self.fc_var1(embed_input1) # [64, H / 2, W / 2]
        mu2 = self.fc_mu2(embed_input2)
        log_var2 = self.fc_var2(embed_input2) # [128, H / 4, W / 4]

        return [mu0, log_var0, mu1, log_var1, mu2, log_var2]
    
    def reparameterize(self, mu0, logvar0, mu1, logvar1, mu2, logvar2):
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std0 = torch.exp(0.5 * logvar0)
        eps0 = torch.randn_like(std0)
        std1 = torch.exp(0.5 * logvar1)
        eps1 = torch.randn_like(std1)
        std2 = torch.exp(0.5 * logvar2)
        eps2 = torch.randn_like(std2)
        return [eps0 * std0 + mu0, eps1 * std1 + mu1, eps2 * std2 + mu2]

class Decoder(nn.Module):
    def __init__(self,
                in_channels: int,
                condition_channels: int) -> None:
        super(Decoder, self).__init__()

        self.in_channels = in_channels
        self.condition_channels = condition_channels

        def ConvT(input_nums, output_nums):
            layer = []
            layer.append(nn.ConvTranspose2d(input_nums, output_nums, kernel_size=(4,4), stride=(2,2), padding=(1,1), bias=False))
            layer.append(nn.BatchNorm2d(output_nums))
            layer.append(nn.ReLU())
            return layer
        
        def Conv(input_nums, output_nums, stride):
            layer = []
            layer.append(nn.Conv2d(input_nums, output_nums, kernel_size=(3,3), stride=stride, padding=(1,1), bias=False))
            layer.append(nn.BatchNorm2d(output_nums))
            layer.append(nn.ReLU())
            return layer
        
        hidden_dims = [64, 128]
        # Build Decoder

        self.decoder_input0 = nn.Sequential(*Conv(hidden_dims[1] * 2, hidden_dims[1] * 2, stride=1))
        self.decoder_input1 = nn.Sequential(*Conv(hidden_dims[0] * 2, hidden_dims[0] * 2, stride=1))
        self.decoder_input2 = nn.Sequential(*Conv(in_channels + condition_channels, in_channels + condition_channels, stride=1))

        self.decconv1 = nn.Sequential(*ConvT(hidden_dims[1] * 2, hidden_dims[1]))
        self.decconv2 = nn.Sequential(*ConvT(hidden_dims[1] * 2, in_channels + condition_channels))

        self.final_layer = nn.Sequential(
                            nn.Conv2d((in_channels + condition_channels) * 2, out_channels= self.in_channels,
                                      kernel_size= 3, padding= 1, bias=False),
                            # nn.Tanh()
                            )
        self.embed_class0 = nn.Conv2d(condition_channels, condition_channels, kernel_size=1)
        self.embed_class1 = nn.Sequential(*Conv(condition_channels, hidden_dims[0], stride=2))
        self.embed_class2 = nn.Sequential(*Conv(hidden_dims[0], hidden_dims[1], stride=2))
    
    def decode(self, z0, z1, z2):
        # z0: [256, H / 4, W / 4] torch.concat([z0, y2])
        result0 = self.decoder_input0(z0) # result0: [256, H / 4, H / 4]
        result0 = self.decconv1(result0) # result0: [128, H / 2, W / 2]
        # z1: [128, H / 2, W / 2] torch.concat([z1, y1])
        result1 = self.decoder_input1(z1) # result1 : [128, H / 2, W / 2]
        result1 = torch.concat([result1, result0], dim = 1) # result1: [256, H / 2, W / 2]
        result1 = self.decconv2(result1) # result1: [34, H, W]
        # z2: [34, H, W]
        result2 = self.decoder_input2(z2) # result2: [34, H, W]
        result2 = torch.concat([result2, result1], dim = 1) # result2: [68, H, W]
        result = self.final_layer(result2)
        return result
    
    def forward(self, z0, z1, z2, y):
        embed_y0 = self.embed_class0(y)
        embed_y1 = self.embed_class1(embed_y0)
        embed_y2 = self.embed_class2(embed_y1)
        
        z0 = torch.cat([z0, embed_y2], dim = 1)
        z1 = torch.cat([z1, embed_y1], dim = 1)
        z2 = torch.cat([z2, embed_y0], dim = 1)
        
        recon = self.decode(z0, z1, z2)
        return recon

def vae_loss(recons, input, mu0, log_var0, mu1, log_var1, mu2, log_var2):

    # kld_weight = 3e-3  # Account for the minibatch samples from the dataset
    # kld_weight = 1024 / 128 / 128 / 31  # Account for the minibatch samples from the dataset
    kld_weight = 1 / 128 / 128  # Account for the minibatch samples from the dataset
    recons_loss =F.l1_loss(recons, input)
    
    Mu0 = rearrange(mu0, 'b c h w -> b (c h w)')
    Log_var0 = rearrange(log_var0, 'b c h w -> b (c h w)')
    kld_weight0 = 1 / Mu0.shape[1]
    Mu1 = rearrange(mu1, 'b c h w -> b (c h w)')
    Log_var1 = rearrange(log_var1, 'b c h w -> b (c h w)')
    kld_weight1 = 1 / Mu1.shape[1]
    Mu2 = rearrange(mu2, 'b c h w -> b (c h w)')
    Log_var2 = rearrange(log_var2, 'b c h w -> b (c h w)')
    kld_weight2 = 1 / Mu2.shape[1]

    kld_loss0 = torch.mean(-0.5 * torch.sum(1 + Log_var0 - Mu0 ** 2 - Log_var0.exp(), dim = 1), dim = 0)
    kld_loss1 = torch.mean(-0.5 * torch.sum(1 + Log_var1 - Mu1 ** 2 - Log_var1.exp(), dim = 1), dim = 0)
    kld_loss2 = torch.mean(-0.5 * torch.sum(1 + Log_var2 - Mu2 ** 2 - Log_var2.exp(), dim = 1), dim = 0)

    loss = recons_loss + kld_weight * (kld_loss0 + kld_loss1 + kld_loss2)
    return loss

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return F.sigmoid(self.model(input))

class vaegan():
    def __init__(self, opt, multiGPU = False) -> None:
        super().__init__()
        self.opt = opt
        self.multiGPU = multiGPU
        self.Encoder = Encoder(31, 3)
        self.G = Decoder(31, 3)
        self.D = NLayerDiscriminator(34)
        if self.multiGPU:
            self.Encoder = nn.DataParallel(self.Encoder)
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)
        self.Encoder.cuda()
        self.G.cuda()
        self.D.cuda()
        self.epoch = 0
        self.end_epoch = opt.end_epoch
        per_epoch_iteration = 1000
        self.total_iteration = per_epoch_iteration*opt.end_epoch
        self.iteration = 0
        self.optimG = optim.RMSprop(self.G.parameters(), lr=self.opt.init_lr, alpha=0.9, eps=1e-8, weight_decay=0,
                                      momentum=0, centered=False)
        self.schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimG, self.total_iteration, eta_min=1e-6)
        self.optimEnc = optim.RMSprop(self.Encoder.parameters(), lr=self.opt.init_lr, alpha=0.9, eps=1e-8, weight_decay=0,
                                      momentum=0, centered=False)
        self.schedulerEnc = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimEnc, self.total_iteration, eta_min=1e-6)
        self.optimD = optim.RMSprop(self.D.parameters(), lr=self.opt.init_lr, alpha=0.9, eps=1e-8, weight_decay=0,
                                      momentum=0, centered=False)
        self.schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimD, self.total_iteration, eta_min=1e-6)
        self.lossl1 = nn.L1Loss()
        self.criterionGAN = nn.MSELoss()
        self.lamda = 1
        self.root = '/work3/s212645/Spectral_Reconstruction/checkpoint/vaegan/'
        if not os.path.exists(self.root):
            os.makedirs(self.root)
    
    def load_dataset(self):
        # load dataset
        print("\nloading dataset ...")
        self.train_data = TrainDataset(data_root=self.opt.data_root, crop_size=self.opt.patch_size, valid_ratio = 0.1, test_ratio=0.1)
        print(f"Iteration per epoch: {len(self.train_data)}")
        self.val_data = ValidDataset(data_root=self.opt.data_root, crop_size=self.opt.patch_size, valid_ratio = 0.1, test_ratio=0.1)
        print("Validation set samples: ", len(self.val_data))
        self.test_data = TestDataset(data_root=self.opt.data_root, crop_size=self.opt.patch_size, valid_ratio = 0.1, test_ratio=0.1)
        print("Validation set samples: ", len(self.val_data))
    
    def vae_recon(self, images, labels):
        [mu0, log_var0, mu1, log_var1, mu2, log_var2] = self.Encoder(labels)
        if self.multiGPU:
            [z2, z1, z0] = self.Encoder.module.reparameterize(mu0, log_var0, mu1, log_var1, mu2, log_var2)
        else:
            [z2, z1, z0] = self.Encoder.reparameterize(mu0, log_var0, mu1, log_var1, mu2, log_var2)
        recon = self.G(z0, z1, z2, images)
        return recon, mu0, log_var0, mu1, log_var1, mu2, log_var2
    
    def G_fake(self, images):
        b, c, h, w = images.shape
        z0 = torch.randn([b, 128, h // 4, w // 4]).cuda()
        z1 = torch.randn([b, 64, h // 2, w // 2]).cuda()
        z2 = torch.randn([b, 31, h, w]).cuda()
        recon = self.G(z0, z1, z2, images)
        return recon
    
    def train(self):
        self.load_dataset()
        record_mrae_loss = 1000
        while self.iteration<self.total_iteration:
            self.G.train()
            self.D.train()
            self.Encoder.train()
            losses = AverageMeter()
            train_loader = DataLoader(dataset=self.train_data, batch_size=self.opt.batch_size, shuffle=True, num_workers=2,
                                    pin_memory=True, drop_last=True)
            val_loader = DataLoader(dataset=self.val_data, batch_size=self.opt.batch_size, shuffle=False, num_workers=2, pin_memory=True)
            for i, (images, labels) in enumerate(train_loader):
                labels = labels.cuda()
                images = images.cuda()
                images = Variable(images)
                labels = Variable(labels)
                
                recon, mu0, log_var0, mu1, log_var1, mu2, log_var2 = self.vae_recon(images, labels)
                fake = self.G_fake(images)
                reconAB = torch.concat([images, recon],dim=1)
                fakeAB = torch.concat([images, fake],dim=1)
                # train D
                # real 
                for p in self.D.parameters():
                    p.requires_grad = True
                self.optimD.zero_grad()
                lrD = self.optimD.param_groups[0]['lr']
                realAB = torch.concat([images, labels], dim=1)
                D_real = self.D(realAB)
                real_labels = torch.ones_like(D_real).cuda()
                loss_real = self.criterionGAN(D_real, real_labels)
                loss_real.backward(retain_graph=True)
                
                # recon
                D_recon = self.D(reconAB.detach())
                fake_labels = torch.zeros_like(D_recon).cuda()
                loss_recon = self.criterionGAN(D_recon, fake_labels)
                loss_recon.backward(retain_graph=True)
                
                # fake
                D_fake = self.D(fakeAB.detach())
                fake_labels = torch.zeros_like(D_fake).cuda()
                loss_fake = self.criterionGAN(D_fake, fake_labels)
                loss_fake.backward(retain_graph=True)
                
                self.optimD.step()
                self.schedulerD.step()
                
                # train G
                self.optimG.zero_grad()
                self.optimEnc.zero_grad()
                lrG = self.optimG.param_groups[0]['lr']
                for p in self.D.parameters():
                    p.requires_grad = False
                
                pred_fake = self.D(fakeAB)
                loss_G_fake = self.criterionGAN(pred_fake, real_labels)
                loss_G_fake.backward(retain_graph=True)
                lossvae = vae_loss(recon, labels, mu0, log_var0, mu1, log_var1, mu2, log_var2)
                lossvae.backward(retain_graph=True)
                pred_recon = self.D(reconAB)
                loss_G_recon = self.criterionGAN(pred_recon, real_labels)
                loss_G_recon.backward(retain_graph=True)
                # train the generator
                self.optimG.step()
                self.schedulerG.step()
                
                # train encoder
                self.optimG.zero_grad()
                self.optimEnc.zero_grad()
                for p in self.D.parameters():
                    p.requires_grad = False
                lossvae = vae_loss(recon, labels, mu0, log_var0, mu1, log_var1, mu2, log_var2)
                lossvae.backward(retain_graph=True)
                self.optimEnc.step()
                self.schedulerEnc.step()
                
                loss_mrae = criterion_mrae(fake, labels)
                losses.update(loss_mrae.data)
                self.iteration = self.iteration+1
                if self.iteration % 20 == 0:
                    print('[iter:%d/%d],lr=%.9f,train_losses.avg=%.9f'
                        % (self.iteration, self.total_iteration, lrG, losses.avg))
            # if self.iteration % 1000 == 0:
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
                
    def validate(self, val_loader):
        self.G.eval()
        self.Encoder.eval()
        losses_mrae = AverageMeter()
        losses_rmse = AverageMeter()
        losses_psnr = AverageMeter()
        losses_sam = AverageMeter()
        losses_sid = AverageMeter()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()
            with torch.no_grad():
                # compute output
                output = self.G_fake(input)
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
        return losses_mrae.avg, losses_rmse.avg, losses_psnr.avg, losses_sam.avg, losses_sid.avg
    
    def test(self, test_loader):
        pass
    
    def save_checkpoint(self, best = False):
        if self.multiGPU:
            state = {
                'epoch': self.epoch,
                'iter': self.iteration,
                'G': self.G.module.state_dict(),
                'D': self.D.module.state_dict(),
                'Enc': self.Encoder.module.state_dict(),
                'optimG': self.optimG.state_dict(),
                'optimD': self.optimD.state_dict(),
                'optimEnc': self.optimEnc.state_dict()
            }
        else:
            state = {
                'epoch': self.epoch,
                'iter': self.iteration,
                'G': self.G.state_dict(),
                'D': self.D.state_dict(),
                'Enc': self.Encoder.state_dict(),
                'optimG': self.optimG.state_dict(),
                'optimD': self.optimD.state_dict(),
                'optimEnc': self.optimEnc.state_dict()
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
            self.Encoder.module.load_state_dict(checkpoint['Enc'])
        else:
            self.G.load_state_dict(checkpoint['G'])
            self.D.load_state_dict(checkpoint['D'])
            self.Encoder.load_state_dict(checkpoint['Enc'])
        self.optimG.load_state_dict(checkpoint['optimG'])
        self.optimD.load_state_dict(checkpoint['optimD'])
        self.optimEnc.load_state_dict(checkpoint['optimEnc'])
        self.iteration = checkpoint['iter']
        self.epoch = checkpoint['epoch']
        print("pretrained model loaded, iteration: ", self.iteration)

if __name__ == '__main__':
    spec = vaegan(opt, multiGPU=opt.multigpu)
    # input = torch.randn([1, 3, 128, 128]).cuda()
    # label = torch.randn([1, 31, 128, 128]).cuda()
    
    # recon, mu0, log_var0, mu1, log_var1, mu2, log_var2 = spec.vae_recon(input, label)
    # print(recon.shape)
    
    # try:
    #     spec.load_checkpoint()
    # except:
    #     print('failed to load trained model')
    spec.train()