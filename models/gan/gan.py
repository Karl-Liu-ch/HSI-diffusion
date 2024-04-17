import sys
sys.path.append('./')
import torch.nn as nn
import torch
import torch.optim as optim
from dataset.datasets import TrainDataset, ValidDataset, TestDataset, TestFullDataset
import os
from torch.utils.data import DataLoader
from torch.autograd import Variable
from models.gan.networks import *
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchsummary import summary
# from models.transformer import MST_Plus_Plus, DTN
from utils import *
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'

# if opt.multigpu:
#     os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
#     local_rank = int(os.environ["LOCAL_RANK"])
#     torch.cuda.set_device(local_rank)
# else:
#     os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# loss function
criterion_mrae = Loss_MRAE()
criterion_rmse = Loss_RMSE()
criterion_psnr = Loss_PSNR()
criterion_psnrrgb = Loss_PSNR()
criterion_sam = Loss_SAM()
criterion_sid = Loss_SID()
criterion_fid = Loss_Fid().cuda()
criterion_ssim = Loss_SSIM().cuda()

def normalize(input):
    return (input - input.min()) / (input.max() - input.min())
    
class Gan():
    def __init__(self, *, 
                 genconfig, 
                 disconfig, 
                 lossconfig,
                 end_epoch, 
                 ckpath, 
                 learning_rate, 
                 data_root, 
                 patch_size, 
                 datanames, 
                 batch_size, 
                 loss_type = 'wasserstein',
                 valid_ratio = 0.1, 
                 test_ratio = 0.1,
                 multigpu = False, 
                 noise = False, 
                 use_feature = True, **kargs):
        super().__init__()
        self.multiGPU = multigpu
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        self.G = instantiate_from_config(genconfig)
        self.D = instantiate_from_config(disconfig)
        self.loss = instantiate_from_config(lossconfig)
        if self.multiGPU:
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)
        self.G.cuda()
        self.D.cuda()
        summary(self.G, (6, 128, 128))
        self.noise = noise
        self.datanames = datanames
        self.epoch = 0
        self.end_epoch = end_epoch
        per_epoch_iteration = 1000
        self.total_iteration = per_epoch_iteration*end_epoch
        self.iteration = 0
        self.best_mrae = 1000
        self.data_root = data_root
        self.patch_size = patch_size
        self.batch_size = batch_size
        
        self.lossl1 = criterion_mrae
        self.loss_min = None
        self.loss_type = loss_type
        self.use_feature = use_feature
        
        self.optimG = optim.Adam(self.G.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        # self.schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimG, self.end_epoch, eta_min=1e-6)
        self.optimD = optim.Adam(self.D.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        # self.schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimD, self.end_epoch, eta_min=1e-6)
        self.root = ckpath
        self.writer = SummaryWriter(log_dir=self.root + 'runs/')
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
    
    
    def load_dataset(self):
        # load dataset
        print("\nloading dataset ...")
        self.train_data = TrainDataset(data_root=self.data_root, crop_size=self.patch_size, valid_ratio = self.valid_ratio, test_ratio=self.test_ratio, datanames = self.datanames)
        print(f"Iteration per epoch: {len(self.train_data)}")
        self.val_data = ValidDataset(data_root=self.data_root, crop_size=self.patch_size, valid_ratio = self.valid_ratio, test_ratio=self.test_ratio, datanames = self.datanames)
        print("Validation set samples: ", len(self.val_data))
        
    def get_last_layer(self):
        if self.multiGPU:
            return self.G.module.get_last_layer()
        else:
            return self.G.get_last_layer()

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
        self.load_dataset()
        iter_per_epoch = len(self.train_data)
        self.total_iteration = self.end_epoch * (iter_per_epoch // self.batch_size + 1)
        self.schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimG, self.total_iteration, eta_min=1e-6)
        self.schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimD, self.total_iteration, eta_min=1e-6)
        while self.epoch<self.end_epoch:
            self.G.train()
            self.D.train()
            losses = AverageMeter()
            train_loader = DataLoader(dataset=self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=32,
                                    pin_memory=True, drop_last=False)
            val_loader = DataLoader(dataset=self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=32, pin_memory=True)
            pbar = tqdm(train_loader)
            for i, batch in enumerate(pbar):
                labels = batch['label'].cuda()
                images = batch['ycrcb'].cuda()
                images = Variable(images)
                labels = Variable(labels)
                if self.noise:
                    z = torch.randn_like(images).cuda()
                    z = torch.concat([z, images], dim=1)
                    z = Variable(z)
                else:
                    z = images
                x_fake = self.G(z)
                
                # train G
                self.optimG.zero_grad()
                lrG = self.optimG.param_groups[0]['lr']
                loss_G = self.loss(self.D, x_fake, labels, images, 0, self.iteration, last_layer = self.get_last_layer())
                loss_G.backward()
                self.optimG.step()
                
                # train D
                self.optimD.zero_grad()
                loss_d = self.loss(self.D, x_fake, labels, images, 1, self.iteration, last_layer = self.get_last_layer())
                loss_d.backward()
                self.optimD.step()
                self.schedulerD.step()
                self.schedulerG.step()
                
                loss_mrae = criterion_mrae(x_fake, labels)
                losses.update(loss_mrae.data)
                self.writer.add_scalar("MRAE/train", loss_mrae, self.iteration)
                self.writer.add_scalar("lr/train", lrG, self.iteration)
                self.writer.add_scalar("loss_G/train", loss_G, self.iteration)
                self.writer.add_scalar("loss_D/train", loss_d, self.iteration)
                self.iteration = self.iteration+1
                logs = {'epoch':self.epoch, 'iter':self.iteration, 'lr':'%.9f'%lrG, 'train_losses':'%.9f'%(losses.avg), 'G_loss':'%.9f'%(loss_G), 'D_loss':'%.9f'%(loss_d)}
                pbar.set_postfix(logs)
            # validation
            mrae_loss, rmse_loss, psnr_loss, sam_loss, sid_loss = self.validate(val_loader)
            print(f'MRAE:{mrae_loss}, RMSE: {rmse_loss}, PNSR:{psnr_loss}, SAM: {sam_loss}, SID: {sid_loss}')
            self.writer.add_scalar("MRAE/val", mrae_loss, self.epoch)
            self.writer.add_scalar("RMSE/val", rmse_loss, self.epoch)
            self.writer.add_scalar("PNSR/val", psnr_loss, self.epoch)
            self.writer.add_scalar("SAM/val", sam_loss, self.epoch)
            # Save model
            print(f'Saving to {self.root}')
            self.save_checkpoint()
            if mrae_loss < self.best_mrae:
                self.best_mrae = mrae_loss
                self.save_checkpoint(True)
            print(" Iter[%06d], Epoch[%06d], learning rate : %.9f, Train MRAE: %.9f, Test MRAE: %.9f, "
                "Test RMSE: %.9f, Test PSNR: %.9f, SAM: %.9f, SID: %.9f " % (self.iteration, 
                                                                self.epoch, lrG, 
                                                                losses.avg, mrae_loss, rmse_loss, psnr_loss, sam_loss, sid_loss))
            self.epoch += 1
            if mrae_loss > 100.0: 
                break

    def hsi2rgb(self, hsi):
        rgb = self.loss.deltaELoss.model_hs2rgb(hsi)
        return rgb

    def validate(self, val_loader):
        self.G.eval()
        losses_mrae = AverageMeter()
        losses_rmse = AverageMeter()
        losses_psnr = AverageMeter()
        losses_sam = AverageMeter()
        losses_sid = AverageMeter()
        pbar = tqdm(val_loader)
        for i, batch in enumerate(pbar):
            target = batch['label'].cuda()
            input = batch['ycrcb'].cuda()
            if self.noise:
                z = torch.randn_like(input).cuda()
                z = torch.concat([z, input], dim=1)
                z = Variable(z)
            else:
                z = input
            with torch.no_grad():
                # compute output
                output = self.G(z)
                if i == 0:
                    rgb = self.hsi2rgb(output)[0,:,:,:]
                    self.writer.add_image("fake/val", rgb, self.epoch)
                    rgb = self.hsi2rgb(target)[0,:,:,:]
                    self.writer.add_image("real/val", rgb, self.epoch)
                loss_mrae = criterion_mrae(output, target)
                loss_rmse = criterion_rmse(output, target)
                loss_psnr = criterion_psnr(output, target)
                loss_sam = criterion_sam(output, target)
                loss_sid = criterion_sid(output, target)
                logs = {'MRAE':'%.9f'%(loss_mrae), 'RMSE':'%.9f'%loss_rmse, 'PSNR':'%.9f'%loss_psnr, 'SAM':'%.9f'%loss_sam}
                pbar.set_postfix(logs)
            # record loss
            losses_mrae.update(loss_mrae.data)
            losses_rmse.update(loss_rmse.data)
            losses_psnr.update(loss_psnr.data)
            losses_sam.update(loss_sam.data)
            losses_sid.update(loss_sid.data)
        criterion_sam.reset()
        criterion_psnr.reset()
        self.metrics['MRAE'][self.epoch]=losses_mrae.avg.cpu().detach().numpy()
        self.metrics['RMSE'][self.epoch]=losses_rmse.avg.cpu().detach().numpy()
        self.metrics['PSNR'][self.epoch]=losses_psnr.avg.cpu().detach().numpy()
        self.metrics['SAM'][self.epoch]=losses_sam.avg.cpu().detach().numpy()
        self.save_metrics()
        return losses_mrae.avg, losses_rmse.avg, losses_psnr.avg, losses_sam.avg, losses_sid.avg
    
    def test(self, modelname, datanames = ['BGU/', 'ARAD/']):
        try: 
            os.mkdir('/work3/s212645/Spectral_Reconstruction/FakeHyperSpectrum/')
            os.mkdir('/work3/s212645/Spectral_Reconstruction/RealHyperSpectrum/')
        except:
            pass
        root = '/work3/s212645/Spectral_Reconstruction/RealHyperSpectrum/'
        # root = '/work3/s212645/Spectral_Reconstruction/RealHyperSpectrum/' + modelname + '/'
        try: 
            os.mkdir('/work3/s212645/Spectral_Reconstruction/FakeHyperSpectrum/' + modelname + '/')
            # os.mkdir('/work3/s212645/Spectral_Reconstruction/RealHyperSpectrum/' + modelname + '/')
        except:
            pass
        test_data = TestDataset(data_root=opt.data_root, crop_size=opt.patch_size, valid_ratio = 0.1, test_ratio=0.1, datanames = datanames)
        print("Test set samples: ", len(test_data))
        test_loader = DataLoader(dataset=test_data, batch_size=opt.batch_size, shuffle=False, num_workers=32, pin_memory=True)
        self.G.eval()
        losses_mrae = AverageMeter()
        losses_rmse = AverageMeter()
        losses_psnr = AverageMeter()
        losses_psnrrgb = AverageMeter()
        losses_sam = AverageMeter()
        losses_sid = AverageMeter()
        losses_fid = AverageMeter()
        losses_ssim = AverageMeter()
        count = 0
        for i, batch in enumerate(tqdm(test_loader)):
            target = batch['label'].cuda()
            input = batch['ycrcb'].cuda()
            rgb_gt = batch['cond'].cuda()
            if self.noise:
                z = torch.randn_like(input).cuda()
                z = torch.concat([z, input], dim=1)
                z = Variable(z)
            else:
                z = input
            with torch.no_grad():
                # compute output
                output = self.G(z)
                rgbs = []
                reals = []
                for j in range(output.shape[0]):
                    mat = {}
                    mat['cube'] = np.transpose(target[j,:,:,:].cpu().numpy(), [1,2,0])
                    mat['rgb'] = np.transpose(rgb_gt[j,:,:,:].cpu().numpy(), [1,2,0])
                    real = mat['rgb']
                    real = (real - real.min()) / (real.max()-real.min())
                    mat['rgb'] = real
                    rgb = SaveSpectral(output[j,:,:,:], count, root='/work3/s212645/Spectral_Reconstruction/FakeHyperSpectrum/' + modelname + '/')
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
                loss_fid = criterion_fid(rgbs, reals)
                loss_ssim = criterion_ssim(rgbs, reals)
                loss_psrnrgb = criterion_psnrrgb(rgbs, reals)
            # record loss
            losses_mrae.update(loss_mrae.data)
            losses_rmse.update(loss_rmse.data)
            losses_psnr.update(loss_psnr.data)
            losses_sam.update(loss_sam.data)
            losses_sid.update(loss_sid.data)
            losses_fid.update(loss_fid.data)
            losses_ssim.update(loss_ssim.data)
            losses_psnrrgb.update(loss_psrnrgb.data)
        criterion_sam.reset()
        criterion_psnr.reset()
        criterion_psnrrgb.reset()
        file = '/zhome/02/b/164706/Master_Courses/thesis/HSI-diffusion/result.txt'
        f = open(file, 'a')
        f.write(modelname+':\n')
        f.write(f'MRAE:{losses_mrae.avg}, RMSE: {losses_rmse.avg}, PNSR:{losses_psnr.avg}, SAM: {losses_sam.avg}, SID: {losses_sid.avg}, FID: {losses_fid.avg}, SSIM: {losses_ssim.avg}, PSNRRGB: {losses_psnrrgb.avg}')
        f.write('\n')
        return losses_mrae.avg, losses_rmse.avg, losses_psnr.avg, losses_sam.avg, losses_sid.avg, losses_fid.avg, losses_ssim.avg, losses_psnrrgb.avg

    def save_metrics(self):
        name = 'metrics.pth'
        torch.save(self.metrics, os.path.join(self.root, name))
        
    def load_metrics(self):
        name = 'metrics.pth'
        checkpoint = torch.load(os.path.join(self.root, name))
        self.metrics = checkpoint
    
    def save_top_k(self, loss):
        save = False
        i = None
        if (loss < self.loss_min).any():
            save = True
            i = np.argwhere((loss < self.loss_min) == True).min()
            self.loss_min[i+1:] = self.loss_min[:-i-1]
            self.loss_min[i] = loss
        return save, i
    
    def save_checkpoint(self, best = False, top_k = None):
        if self.multiGPU:
            state = {
                'epoch': self.epoch,
                'iter': self.iteration,
                'best_mrae': self.best_mrae,
                'loss_min': self.loss_min, 
                'G': self.G.module.state_dict(),
                'D': self.D.module.state_dict(),
                'optimG': self.optimG.state_dict(),
                'optimD': self.optimD.state_dict(),
            }
        else:
            state = {
                'epoch': self.epoch,
                'iter': self.iteration,
                'best_mrae': self.best_mrae,
                'loss_min': self.loss_min, 
                'G': self.G.state_dict(),
                'D': self.D.state_dict(),
                'optimG': self.optimG.state_dict(),
                'optimD': self.optimD.state_dict(),
            }
        if best: 
            name = 'net_epoch_best.pth'
            torch.save(state, os.path.join(self.root, name))
        if top_k is not None:
            name = f'net_epoch_{top_k}.pth'
            torch.save(state, os.path.join(self.root, name))
            print(f'top: {top_k} saved. ')
        name = 'net.pth'
        torch.save(state, os.path.join(self.root, name))
        
    def load_checkpoint(self, best = False):
        if best:
            checkpoint = torch.load(os.path.join(self.root, 'net_epoch_best.pth'))
        else:
            checkpoint = torch.load(os.path.join(self.root, 'net.pth'))
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
        self.best_mrae = checkpoint['best_mrae']
        try:
            self.load_metrics()
        except:
            pass
        print("pretrained model loaded, iteration: ", self.iteration)

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
        test_data = TestFullDataset(data_root=self.data_root, crop_size=self.patch_size, valid_ratio = 0.1, test_ratio=0.1)
        print("Test set samples: ", len(test_data))
        test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False, num_workers=32, pin_memory=True)
        count = 0
        for i, (input, target, rgb_gt) in enumerate(tqdm(test_loader)):
            input = input.cuda()
            target = target.cuda()
            rgb_gt = rgb_gt.cuda()
            H, W = input.shape[-2], input.shape[-1]
            # if modelname == 'SNCWGANDTN' and (H != H_ or W != W_):
            #     self.G = DTN(in_dim=6, 
            #             out_dim=31,
            #             img_size=[H, W], 
            #             window_size=8, 
            #             n_block=[2,2,2,2], 
            #             bottleblock = 4).to(device)
            #     H_ = H
            #     W_ = W
            #     self.load_checkpoint(best=True)
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
    # input = torch.randn([1, 6, 128, 128]).cuda()
    # D = Discriminator(31).cuda()
    # G = Generator(6, 31).cuda()
    
    # output = G(input)
    # print(output.shape)
    
    spec = Gan(opt, multiGPU=opt.multigpu)
    # spec.load_checkpoint()
    spec.train()