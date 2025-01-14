import sys
sys.path.append('./')
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.optim as optim
import torch.backends.cudnn as cudnn
from utils import *
import os
from torch.utils.data import DataLoader
from torch.autograd import Variable
import lightning as pl
from utils import instantiate_from_config
from omegaconf import OmegaConf
from options import opt
import numpy as np
from timm.scheduler.cosine_lr import CosineLRScheduler
from differential_color_functions_no_device import rgb2lab_diff as rgb2lab_diff_cuda
from differential_color_functions_no_device import ciede2000_diff as ciede2000_diff_cuda

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# loss function
criterion_mrae = Loss_MRAE()
criterion_rmse = Loss_RMSE()
criterion_psnr = Loss_PSNR()
criterion_psnrrgb = Loss_PSNR()
criterion_sam = SAMLoss()

def deltaE_Loss(fake, gt):
    loss = ciede2000_diff_cuda(rgb2lab_diff_cuda(gt), rgb2lab_diff_cuda(fake))
    color_loss=loss.mean()
    return color_loss

class Loss_DeltaE(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = device
        self.model_hs2rgb = nn.Conv2d(31, 3, 1, bias=False)
        cie_matrix = CAM_FILTER
        cie_matrix = torch.from_numpy(np.transpose(cie_matrix, [1, 0])).unsqueeze(-1).unsqueeze(-1).float()
        self.model_hs2rgb.weight.data = cie_matrix
        self.model_hs2rgb.weight.requires_grad = False

    def forward(self, outputs, label, rgb_gt = None):
        # hs2rgb
        if rgb_gt is None:
            rgb_tensor = self.model_hs2rgb(outputs)
            rgb_tensor = normalization_image(rgb_tensor)
            rgb_label = self.model_hs2rgb(label)
            rgb_label = normalization_image(rgb_label)
        else:
            rgb_tensor = self.model_hs2rgb(outputs)
            rgb_tensor = normalization_image(rgb_tensor)
            rgb_label = normalization_image(rgb_gt)
        deltaE = deltaE_Loss(rgb_tensor, rgb_label)
        return deltaE

# criterion_deltae = LossDeltaE().cuda()
# deltaE_criterion = Loss_DeltaE()

class BaseModel(pl.LightningModule):
    def __init__(self, *,
                 epochs,
                 learning_rate,
                 cond_key, 
                 modelconfig,
                 num_warmup = 0,
                 monitor = None, 
                 finetune = False,
                 **kwargs, 
                 ):
        super().__init__()
        self.num_warmup = num_warmup
        self.automatic_optimization = False
        self.opt = opt
        self.epoch = 0
        self.end_epoch = epochs
        self.iteration = 0
        self.model = instantiate_from_config(modelconfig)
        self.best_mrae = 1000
        self._temp_epoch = 0
        self._optimizer_states = None
        self.name = modelconfig.target.split('.')[-1]
        self.criterion = criterion_mrae
        self.root = '/work3/s212645/Spectral_Reconstruction/checkpoint/'+self.name+'/'
        self.learning_rate = learning_rate
        self.deltaE_criterion = Loss_DeltaE()
        print(learning_rate)
        self.cond_key = cond_key
        self.finetune = finetune
        if monitor is not None:
            self.monitor = monitor
        # make checkpoint dir
        # if not os.path.exists(opt.outf):
        #     os.makedirs(opt.outf)
        if not os.path.exists(self.root):
            os.makedirs(self.root)

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
        if self.finetune:
            self._temp_epoch = 0
            self._temp_global_step = 0
        else:
            self._temp_epoch = torch.load(path, map_location="cpu")['epoch']
            self._temp_global_step = torch.load(path, map_location="cpu")['global_step']
        print(f"Restored from {path}")
        self.init_optim_ckpt(path)

    def init_optim_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["optimizer_states"]
        self._optimizer_states = sd
        print(f"Restored optimizer from {path}")

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        sch = self.lr_schedulers()
        images = batch[self.cond_key]
        labels = batch['label']
        images = Variable(images)
        labels = Variable(labels)

        self.toggle_optimizer(opt)
        output = self.model(images)
        loss_mrae = criterion_mrae(output, labels)
        loss_sam = criterion_sam(output, labels)
        loss_deltaE = self.deltaE_criterion(output, labels)
        # loss_deltaE = deltaE_criterion(output, labels)
        loss_G = loss_mrae + loss_sam * 0.1 + loss_deltaE * 0.1
        log_dict_g = {'train/mrae': loss_mrae}
        self.log_dict(log_dict_g, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict({"lr": opt.param_groups[0]['lr']}, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.manual_backward(loss_G)
        opt.step()
        opt.zero_grad()
        self.untoggle_optimizer(opt)
        sch.step_update(self.global_step)
        return loss_mrae

    def validation_step(self, batch, batch_idx):
        images = batch[self.cond_key]
        labels = batch['label']
        images = Variable(images)
        labels = Variable(labels)
        output = self.model(images)
        loss_mrae = criterion_mrae(output, labels).detach()
        loss_rmse = criterion_rmse(output, labels).detach()
        loss_psnr = criterion_psnr(output, labels).detach()
        loss_sam = criterion_sam(output, labels).detach()
        # criterion_sam.reset()
        criterion_psnr.reset()
        self.log_dict({'val/mrae': loss_mrae}, sync_dist=True, prog_bar=True, on_epoch=True, on_step=True)
        self.log_dict({'val/rmse': loss_rmse, 'val/psnr': loss_psnr, 'val/sam': loss_sam}, sync_dist=True)
    
    def configure_optimizers(self):
        self.trainer.fit_loop.setup_data()
        self.iter_per_epoch = len(self.trainer.train_dataloader)
        iter_per_epoch = self.iter_per_epoch
        print(f'epoch iterations: {iter_per_epoch}')
        print(f'total iterations: {self.end_epoch * iter_per_epoch}')
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(self.model.parameters(),
                                  lr=lr, betas=(0.9, 0.999))
        warmup_lr_init = lr
        sch_ae = CosineLRScheduler(opt_ae,t_initial=(self.end_epoch - self.num_warmup) * iter_per_epoch,
                                            cycle_mul = 1,cycle_decay = 1,lr_min=1e-6,
                                            warmup_lr_init=warmup_lr_init,warmup_t=self.num_warmup * iter_per_epoch,
                                            cycle_limit=1,t_in_epochs=False)
        if self._optimizer_states is not None:
            try:
                opt_ae.load_state_dict(self._optimizer_states[0])
                for group in opt_ae.param_groups:
                    group['lr'] = lr
            except:
                pass
        return({"optimizer": opt_ae, "lr_scheduler": sch_ae})
    
    # def on_train_start(self):
    #     self.iter_per_epoch = len(self.trainer.train_dataloader)
    #     self.set_current_epoch(self._temp_epoch) # This is being loaded from the model
    #     total_batch_idx = self.current_epoch * len(self.trainer.train_dataloader)
    #     global_step = total_batch_idx
    #     self.set_global_step(global_step)
    #     self.iterations = self.global_step
    #     print(self.current_epoch, self.global_step)

    def on_train_start(self) -> None:
        self.iter_per_epoch = len(self.trainer.train_dataloader)
        self.set_current_epoch(self._temp_epoch) # This is being loaded from the model
        self.set_global_step(self._temp_global_step)
        # self.global_step_manually = self.current_epoch * self.iter_per_epoch
        # self.modify_iter = self.global_step_manually - self.global_step
        # print(self.global_step_manually)
        print(self.epoch, self.global_step)

    def set_current_epoch(self, epoch: int):
        self.trainer.fit_loop.epoch_progress.current.processed = epoch
        self.trainer.fit_loop.epoch_progress.current.completed = epoch
        assert self.current_epoch == epoch, f"{self.current_epoch} != {epoch}"

    def set_global_step(self, global_step: int):
        self.trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.total.completed = (
            global_step
        )
        self.trainer.fit_loop.epoch_loop.automatic_optimization.optim_progress.optimizer.step.total.completed = (
            global_step
        )
        assert self.global_step == global_step, f"{self.global_step} != {global_step}"

    def set_total_batch_idx(self, total_batch_idx: int):
        self.trainer.fit_loop.epoch_loop.batch_progress.total.ready = (
            total_batch_idx + 1
        )
        self.trainer.fit_loop.epoch_loop.batch_progress.total.completed = (
            total_batch_idx
        )
        assert (
            self.total_batch_idx == total_batch_idx + 1
        ), f"{self.total_batch_idx} != {total_batch_idx + 1}"

    @property
    def total_batch_idx(self) -> int:
        return self.trainer.fit_loop.epoch_loop.total_batch_idx + 1
