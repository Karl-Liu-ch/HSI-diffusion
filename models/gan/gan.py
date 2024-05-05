import sys
sys.path.append('./')
import torch.nn as nn
import torch
import torch.optim as optim
from dataset.datasets import *
import os
from torch.utils.data import DataLoader
from torch.autograd import Variable
from models.gan.networks import *
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchsummary import summary
import shutil
# from models.transformer import MST_Plus_Plus, DTN
from utils import *
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
from timm.scheduler.cosine_lr import CosineLRScheduler

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

def normalize(cond):
    return (cond - input.min()) / (input.max() - input.min())
    
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
                 image_key = 'label',
                 cond_key = 'ycrcb',
                 loss_type = 'wasserstein',
                 valid_ratio = 0.1, 
                 test_ratio = 0.1,
                 n_critic = 2,
                 multigpu = False, 
                 noise = False, 
                 use_feature = True,
                 progressive_train = True,
                 num_warmup = 0,
                 patience = 10,
                   **kargs):
        super().__init__()
        self.earlystop = EarlyStopper(patience=patience, min_delta=1e-2, start_epoch=10, gl_weight=1.4)
        self.progressive_module = EarlyStopper(patience=patience, min_delta=1e-2, start_epoch=10, gl_weight=1.4)
        self.image_key = image_key
        self.cond_key = cond_key
        self.n_critic = n_critic
        self.multiGPU = multigpu
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        self.progressive_train = progressive_train
        self.G = instantiate_from_config(genconfig)
        self.D = instantiate_from_config(disconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.G.apply(init_weights_uniform)
        if self.multiGPU:
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)
        self.G.cuda()
        self.D.cuda()
        if cond_key == 'ycrcb':
            summary(self.G, (6, 128, 128))
            summary(self.D, (37, 128, 128))
        elif cond_key == 'cond':
            summary(self.G, (3, 128, 128))
            summary(self.D, (34, 128, 128))
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
        self.top_k = 3
        self.num_warmup = num_warmup
        
        self.lossl1 = criterion_mrae
        self.loss_min = np.ones([self.top_k]) * 1000
        self.loss_type = loss_type
        self.use_feature = use_feature
        
        # self.optimG = optim.Adam(self.G.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)
        # self.optimD = optim.Adam(self.D.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)
        # self.optimG = optim.AdamW(self.G.parameters(), lr=learning_rate, betas=(0.5, 0.9), eps=1e-8, weight_decay=0.05)
        self.optimG = optim.Adam(self.G.parameters(), lr=learning_rate, betas=(0.5, 0.9), eps=1e-14)
        self.optimD = optim.Adam(self.D.parameters(), lr=learning_rate, betas=(0.5, 0.9), eps=1e-12)
        self.schedulerG = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimG, T_0=self.total_iteration, T_mult=1, eta_min=1e-6, last_epoch=-1)
        self.schedulerD = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimD, T_0=self.total_iteration, T_mult=1, eta_min=1e-6, last_epoch=-1)
        self.learning_rate = learning_rate
        self.root = ckpath
        if not opt.resume:
            shutil.rmtree(self.root + 'runs/', ignore_errors=True)
            # os.rmdir(self.root + 'runs/')
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
        self.train_data = TrainDataset(data_root=self.data_root, crop_size=self.patch_size, valid_ratio = self.valid_ratio, test_ratio=self.test_ratio, datanames = self.datanames, stride=128)
        print(f"Iteration per epoch: {len(self.train_data)}")
        self.val_data = ValidDataset(data_root=self.data_root, crop_size=1e8, valid_ratio = self.valid_ratio, test_ratio=self.test_ratio, datanames = self.datanames)
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
    
    def get_input(self, batch):
        label = Variable(batch[self.image_key].to(device))
        cond = Variable(batch[self.cond_key].to(device))
        return label, cond

    def train(self):
        self.load_dataset()
        train_loader = DataLoader(dataset=self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=8,
                                pin_memory=True, drop_last=False)
        iter_per_epoch = len(train_loader)
        self.total_iteration = self.end_epoch * (iter_per_epoch)
        self.schedulerG = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimG, T_0=self.total_iteration, T_mult=1, eta_min=1e-6, last_epoch=self.iteration)
        self.schedulerD = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimD, T_0=self.total_iteration, T_mult=1, eta_min=1e-6, last_epoch=self.iteration)
        # self.schedulerG = CosineLRScheduler(self.optimG,t_initial=(self.end_epoch - self.num_warmup) * iter_per_epoch,
        #                                     cycle_mul = 1,cycle_decay = 1,lr_min=1e-6,
        #                                     warmup_lr_init=self.learning_rate,warmup_t=self.num_warmup * iter_per_epoch,
        #                                     cycle_limit=1,t_in_epochs=False)
        # self.schedulerD = CosineLRScheduler(self.optimD,t_initial=(self.end_epoch - self.num_warmup) * iter_per_epoch,
        #                                     cycle_mul = 1,cycle_decay = 1,lr_min=1e-6,
        #                                     warmup_lr_init=self.learning_rate,warmup_t=self.num_warmup * iter_per_epoch,
        #                                     cycle_limit=1,t_in_epochs=False)
        # self.schedulerG = CosineLRScheduler(self.optimG,t_initial=10*iter_per_epoch,cycle_mul = 2.5,cycle_decay = 0.5,
        #                                     lr_min=1e-6,warmup_lr_init=self.learning_rate,
        #                                     warmup_t=self.num_warmup*iter_per_epoch,warmup_prefix=False,
        #                                     cycle_limit=3,t_in_epochs=False,)
        # self.schedulerD = CosineLRScheduler(self.optimD,t_initial=10*iter_per_epoch,cycle_mul = 2.5,cycle_decay = 0.5,
        #                                     lr_min=1e-6,warmup_lr_init=self.learning_rate,
        #                                     warmup_t=self.num_warmup*iter_per_epoch,warmup_prefix=False,
        #                                     cycle_limit=3,t_in_epochs=False,)
        while self.epoch<self.end_epoch:
            self.G.train()
            self.D.train()
            losses = AverageMeter()
            g_losses = AverageMeter()
            d_losses = AverageMeter()
            train_loader = DataLoader(dataset=self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=8,
                                    pin_memory=True, drop_last=False)
            if len(self.val_data) > 95:
                val_loader = DataLoader(dataset=self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=8, pin_memory=True)
            else:
                val_loader = DataLoader(dataset=self.val_data, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
            pbar = tqdm(train_loader)
            for i, batch in enumerate(pbar):
                label, cond = self.get_input(batch)
                if self.noise:
                    z = torch.randn_like(cond).cuda()
                    z = torch.concat([z, cond], dim=1)
                    z = Variable(z)
                else:
                    z = cond
                    
                x_fake = self.G(z)

                if self.iteration % self.n_critic == 0:
                    # train D
                    # for _ in range(self.n_critic):
                    self.optimD.zero_grad()
                    loss_d, log_d = self.loss(self.D, x_fake, label, cond, self.epoch, mode = 'dics', last_layer = self.get_last_layer())
                    loss_d.backward()
                    self.optimD.step()
                
                # train G
                # for _ in range(self.n_critic):
                # x_fake = self.G(z)
                self.optimG.zero_grad()
                lrG = self.optimG.param_groups[0]['lr']
                loss_G, log_g = self.loss(self.D, x_fake, label, cond, self.epoch, mode = 'gen', last_layer = self.get_last_layer())
                loss_G.backward()
                self.optimG.step()
                
                if isinstance(self.schedulerD, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                    self.schedulerD.step()
                    self.schedulerG.step()
                elif isinstance(self.schedulerD, CosineLRScheduler):
                    self.schedulerD.step_update(self.iteration)
                    self.schedulerG.step_update(self.iteration)
                
                loss_mrae = criterion_mrae(x_fake, label)
                losses.update(loss_mrae.data)
                g_losses.update(loss_G.data)
                d_losses.update(loss_d.data)
                self.writer.add_scalar("MRAE/train", loss_mrae, self.iteration)
                self.writer.add_scalar("lr/train", lrG, self.iteration)
                self.writer.add_scalar("loss_G/train", loss_G, self.iteration)
                self.writer.add_scalar("loss_D/train", loss_d, self.iteration)
                self.writer.add_scalar("gen loss/train", log_g['gen loss'], self.iteration)
                self.writer.add_scalar("delta e loss/train", log_g['delta e'], self.iteration)
                self.writer.add_scalar("disc loss real/train", log_d['real loss'], self.iteration)
                self.writer.add_scalar("disc loss fake/train", log_d['fake loss'], self.iteration)
                self.iteration = self.iteration+1
                logs = {'epoch':self.epoch, 'iter':self.iteration, 'lr':'%.9f'%lrG, 'train_losses':'%.9f'%(losses.avg), 'G_loss':'%.9f'%(loss_G), 'D_loss':'%.9f'%(loss_d)}
                pbar.set_postfix(logs)
            # validation
            mrae_loss, rmse_loss, psnr_loss, sam_loss, sid_loss = self.validate(val_loader)
            print(f'MRAE:{mrae_loss}, RMSE: {rmse_loss}, PSNR:{psnr_loss}, SAM: {sam_loss}, SID: {sid_loss}')
            self.writer.add_scalar("MRAE/val", mrae_loss, self.epoch)
            self.writer.add_scalar("RMSE/val", rmse_loss, self.epoch)
            self.writer.add_scalar("PSNR/val", psnr_loss, self.epoch)
            self.writer.add_scalar("SAM/val", sam_loss, self.epoch)
            # Save model
            print(f'Saving to {self.root}')
            self.save_checkpoint()
            if mrae_loss < self.best_mrae:
                self.best_mrae = mrae_loss
                self.save_checkpoint(True)
            save_top, top_n = self.save_top_k(mrae_loss.detach().cpu().numpy())
            if save_top:
                self.save_checkpoint(top_k=top_n)
            print(" Iter[%06d], Epoch[%06d], learning rate : %.9f, Train MRAE: %.9f, Test MRAE: %.9f, "
                "Test RMSE: %.9f, Test PSNR: %.9f, SAM: %.9f, SID: %.9f " % (self.iteration, 
                                                                self.epoch, lrG, 
                                                                losses.avg, mrae_loss, rmse_loss, psnr_loss, sam_loss, sid_loss))
            if self.progressive_module.early_stop(mrae_loss, losses.avg) and self.progressive_train and  self.patch_size < 512:
                self.progressive_training()
                self.progressive_module.reset()
                self.earlystop.reset()
            if self.earlystop.early_stop(mrae_loss, losses.avg):
                print('early stopped because no improvement in validation')
                break
            if mrae_loss > 100.0: 
                break
            self.epoch += 1

    def progressive_training(self):
        self.optimG = optim.Adam(self.G.parameters(), lr=4e-5, betas=(0.5, 0.9), eps=1e-14)
        self.optimD = optim.Adam(self.D.parameters(), lr=4e-5, betas=(0.5, 0.9), eps=1e-12)
        self.schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimG, self.total_iteration, eta_min=1e-6)
        self.schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimD, self.total_iteration, eta_min=1e-6)
        self.patch_size = self.patch_size * 4
        if self.patch_size > 482:
            arg = False
        else:
            arg = True
        self.train_data = TrainDataset(data_root=self.data_root, crop_size=self.patch_size, valid_ratio = 0.1, test_ratio=0.1, arg=arg, datanames = self.datanames, stride=self.patch_size)
        print(f"Iteration per epoch: {len(self.train_data)}")
        self.batch_size = 1

    def finetuning(self):
        self.patch_size = 1e8
        self.batch_size = 1
        self.load_dataset()
        self.load_checkpoint(best=True)
        self.learning_rate = 4e-5
        self.ft_end_epoch = 10
        self.ft_start_iteration = self.iteration
        train_loader = DataLoader(dataset=self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=8,
                                pin_memory=True, drop_last=False)
        iter_per_epoch = len(train_loader)
        self.total_iteration = self.end_epoch * (iter_per_epoch)
        self.schedulerG = CosineLRScheduler(self.optimG,t_initial=self.ft_end_epoch * iter_per_epoch,
                                            cycle_mul = 1,cycle_decay = 1,lr_min=1e-6,
                                            warmup_lr_init=self.learning_rate,warmup_t=0,
                                            cycle_limit=1,t_in_epochs=False)
        self.schedulerD = CosineLRScheduler(self.optimD,t_initial=self.ft_end_epoch * iter_per_epoch,
                                            cycle_mul = 1,cycle_decay = 1,lr_min=1e-6,
                                            warmup_lr_init=self.learning_rate,warmup_t=0,
                                            cycle_limit=1,t_in_epochs=False)
        while self.epoch<self.ft_end_epoch + self.end_epoch:
            self.G.train()
            self.D.train()
            losses = AverageMeter()
            g_losses = AverageMeter()
            d_losses = AverageMeter()
            train_loader = DataLoader(dataset=self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=8,
                                    pin_memory=True, drop_last=False)
            if len(self.val_data) > 95:
                val_loader = DataLoader(dataset=self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=8, pin_memory=True)
            else:
                val_loader = DataLoader(dataset=self.val_data, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
            pbar = tqdm(train_loader)
            for i, batch in enumerate(pbar):
                label, cond = self.get_input(batch)
                if self.noise:
                    z = torch.randn_like(cond).cuda()
                    z = torch.concat([z, cond], dim=1)
                    z = Variable(z)
                else:
                    z = cond
                    
                x_fake = self.G(z)

                if self.iteration % self.n_critic == 0:
                    # train D
                    # for _ in range(self.n_critic):
                    self.optimD.zero_grad()
                    loss_d, log_d = self.loss(self.D, x_fake, label, cond, self.epoch, mode = 'dics', last_layer = self.get_last_layer())
                    loss_d.backward()
                    self.optimD.step()
                
                # train G
                # for _ in range(self.n_critic):
                # x_fake = self.G(z)
                self.optimG.zero_grad()
                lrG = self.optimG.param_groups[0]['lr']
                loss_G, log_g = self.loss(self.D, x_fake, label, cond, self.epoch, mode = 'gen', last_layer = self.get_last_layer())
                loss_G.backward()
                self.optimG.step()
                
                if isinstance(self.schedulerD, CosineLRScheduler):
                    self.schedulerD.step_update(self.iteration - self.ft_start_iteration)
                    self.schedulerG.step_update(self.iteration - self.ft_start_iteration)
                
                loss_mrae = criterion_mrae(x_fake, label)
                losses.update(loss_mrae.data)
                g_losses.update(loss_G.data)
                d_losses.update(loss_d.data)
                self.writer.add_scalar("MRAE/train", loss_mrae, self.iteration)
                self.writer.add_scalar("lr/train", lrG, self.iteration)
                self.writer.add_scalar("loss_G/train", loss_G, self.iteration)
                self.writer.add_scalar("loss_D/train", loss_d, self.iteration)
                self.writer.add_scalar("gen loss/train", log_g['gen loss'], self.iteration)
                self.writer.add_scalar("delta e loss/train", log_g['delta e'], self.iteration)
                self.writer.add_scalar("disc loss real/train", log_d['real loss'], self.iteration)
                self.writer.add_scalar("disc loss fake/train", log_d['fake loss'], self.iteration)
                self.iteration = self.iteration+1
                logs = {'epoch':self.epoch, 'iter':self.iteration, 'lr':'%.9f'%lrG, 'train_losses':'%.9f'%(losses.avg), 'G_loss':'%.9f'%(loss_G), 'D_loss':'%.9f'%(loss_d)}
                pbar.set_postfix(logs)
            # validation
            mrae_loss, rmse_loss, psnr_loss, sam_loss, sid_loss = self.validate(val_loader)
            print(f'MRAE:{mrae_loss}, RMSE: {rmse_loss}, PSNR:{psnr_loss}, SAM: {sam_loss}, SID: {sid_loss}')
            self.writer.add_scalar("MRAE/val", mrae_loss, self.epoch)
            self.writer.add_scalar("RMSE/val", rmse_loss, self.epoch)
            self.writer.add_scalar("PSNR/val", psnr_loss, self.epoch)
            self.writer.add_scalar("SAM/val", sam_loss, self.epoch)
            # Save model
            print(f'Saving to {self.root}')
            self.save_checkpoint()
            if mrae_loss < self.best_mrae:
                self.best_mrae = mrae_loss
                self.save_checkpoint(True)
            save_top, top_n = self.save_top_k(mrae_loss.detach().cpu().numpy())
            if save_top:
                self.save_checkpoint(top_k=top_n)
            print(" Iter[%06d], Epoch[%06d], learning rate : %.9f, Train MRAE: %.9f, Test MRAE: %.9f, "
                "Test RMSE: %.9f, Test PSNR: %.9f, SAM: %.9f, SID: %.9f " % (self.iteration, 
                                                                self.epoch, lrG, 
                                                                losses.avg, mrae_loss, rmse_loss, psnr_loss, sam_loss, sid_loss))
            if self.progressive_module.early_stop(mrae_loss, losses.avg) and self.progressive_train and  self.patch_size < 512:
                self.progressive_training()
                self.progressive_module.reset()
                self.earlystop.reset()
            if self.earlystop.early_stop(mrae_loss, losses.avg):
                print('early stopped because no improvement in validation')
                break
            if mrae_loss > 100.0: 
                break
            self.epoch += 1


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
            label, cond = self.get_input(batch)
            if self.noise:
                z = torch.randn_like(cond).cuda()
                z = torch.concat([z, cond], dim=1)
                z = Variable(z)
            else:
                z = cond
            with torch.no_grad():
                # compute output
                output = self.G(z)
                if i == 0:
                    rgb = self.hsi2rgb(output)[0,:,:,:]
                    self.writer.add_image("fake/val", rgb, self.epoch)
                    rgb = self.hsi2rgb(label)[0,:,:,:]
                    self.writer.add_image("real/val", rgb, self.epoch)
                loss_mrae = criterion_mrae(output, label)
                loss_rmse = criterion_rmse(output, label)
                loss_psnr = criterion_psnr(output, label)
                loss_sam = criterion_sam(output, label)
                loss_sid = criterion_sid(output, label)
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
    
    def test(self, modelname):
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
        test_data = TestDataset(data_root=self.data_root, crop_size=1e8, valid_ratio = 0.1, test_ratio=0.1, datanames = self.datanames)
        print("Test set samples: ", len(test_data))
        test_loader = DataLoader(dataset=test_data, batch_size=2, shuffle=False, num_workers=32, pin_memory=True)
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
            label, cond = self.get_input(batch)
            rgb_gt = batch['cond'].cuda()
            if self.noise:
                z = torch.randn_like(cond).cuda()
                z = torch.concat([z, cond], dim=1)
                z = Variable(z)
            else:
                z = cond
            with torch.no_grad():
                # compute output
                output = self.G(z)
                rgbs = []
                reals = []
                for j in range(output.shape[0]):
                    mat = {}
                    mat['cube'] = np.transpose(label[j,:,:,:].cpu().numpy(), [1,2,0])
                    mat['rgb'] = np.transpose(rgb_gt[j,:,:,:].cpu().numpy(), [1,2,0])
                    real = mat['rgb']
                    real = (real - real.min()) / (real.max()-real.min())
                    mat['rgb'] = real
                    rgb = SaveSpectral(output[j,:,:,:], count, root='/work3/s212645/Spectral_Reconstruction/FakeHyperSpectrum/' + modelname + '/')
                    rgbs.append(rgb)
                    reals.append(real)
                    count += 1
                loss_mrae = criterion_mrae(output, label)
                loss_rmse = criterion_rmse(output, label)
                loss_psnr = criterion_psnr(output, label)
                loss_sam = criterion_sam(output, label)
                loss_sid = criterion_sid(output, label)
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
        f.write(f'MRAE:{losses_mrae.avg}, RMSE: {losses_rmse.avg}, PSNR:{losses_psnr.avg}, SAM: {losses_sam.avg}, SID: {losses_sid.avg}, FID: {losses_fid.avg}, SSIM: {losses_ssim.avg}, PSNRRGB: {losses_psnrrgb.avg}')
        f.write('\n')
        f.close()
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
                'schedulerG': self.schedulerG.state_dict(),
                'schedulerD': self.schedulerD.state_dict(),
                'early_stop':self.earlystop.save(),
                'progressive_module': self.progressive_module.save(),
                'patch_size': self.patch_size,
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
                'schedulerG': self.schedulerG.state_dict(),
                'schedulerD': self.schedulerD.state_dict(),
                'early_stop':self.earlystop.save(),
                'progressive_module': self.progressive_module.save(),
                'patch_size': self.patch_size,
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
        self.loss_min = checkpoint['loss_min']
        self.patch_size = checkpoint['patch_size']
        self.earlystop.load(checkpoint['early_stop'])
        self.progressive_module.load(checkpoint['progressive_module'])
        self.schedulerG.load_state_dict(checkpoint['schedulerG'])
        self.schedulerD.load_state_dict(checkpoint['schedulerD'])
        try:
            self.load_metrics()
        except:
            pass
        print("pretrained model loaded, iteration: ", self.iteration)

    def test_full_resol(self, modelname, test_loaders):
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
        count = 0
        for test_loader, name in zip(test_loaders, ['ARAD', 'BGU', 'CAVE']):
            for i, batch in enumerate(tqdm(test_loader)):
                save_path = f'/work3/s212645/Spectral_Reconstruction/FakeHyperSpectrum/{modelname}-{name}/'
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                label, cond = self.get_input(batch)
                rgb_gt = batch['cond'].cuda()
                if self.noise:
                    z = torch.randn_like(cond).cuda()
                    z = torch.concat([z, cond], dim=1)
                    z = Variable(z)
                else:
                    z = cond
                with torch.no_grad():
                    output = self.G(z)
                    rgbs = []
                    reals = []
                    for j in range(output.shape[0]):
                        mat = {}
                        mat['cube'] = np.transpose(label[j,:,:,:].cpu().numpy(), [1,2,0])
                        mat['rgb'] = np.transpose(rgb_gt[j,:,:,:].cpu().numpy(), [1,2,0])
                        real = np.transpose(rgb_gt[j,:,:,:].cpu().numpy(), [1,2,0])
                        real = (real - real.min()) / (real.max()-real.min())
                        mat['rgb'] = real
                        rgb = SaveSpectral(output[j,:,:,:], count, root=save_path)
                        rgbs.append(rgb)
                        reals.append(real)
                        count += 1
                    loss_mrae = criterion_mrae(output, label)
                    loss_rmse = criterion_rmse(output, label)
                    loss_psnr = criterion_psnr(output, label)
                    loss_sam = criterion_sam(output, label)
                    loss_sid = criterion_sid(output, label)
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
            f.write(f'{modelname}-{name}:\n')
            f.write(f'MRAE:{losses_mrae.avg}, RMSE: {losses_rmse.avg}, PSNR:{losses_psnr.avg}, SAM: {losses_sam.avg}, SID: {losses_sid.avg}, SSIM: {losses_ssim.avg}, PSNRRGB: {losses_psnrrgb.avg}')
            f.write('\n')
            f.close()

if __name__ == '__main__':
    # cond = torch.randn([1, 6, 128, 128]).cuda()
    # D = Discriminator(31).cuda()
    # G = Generator(6, 31).cuda()
    
    # output = G(cond)
    # print(output.shape)
    
    spec = Gan(opt, multiGPU=opt.multigpu)
    # spec.load_checkpoint()
    spec.train()