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
import math
import re
from dataset.datasets import TestFullDataset
from models.gan.Basemodel import BaseModel, criterion_mrae, AverageMeter, SAM
from utils import *
from models.gan.Utils import Log_loss, Itself_loss
from models.transformer.MST_Plus_Plus import MST_Plus_Plus
from color_loss import deltaELoss
import pytorch_lightning as pl
import lightning as l
from pytorch_lightning.trainer import Trainer
from utils import instantiate_from_config
from omegaconf import OmegaConf
from timm.scheduler.cosine_lr import CosineLRScheduler
config_path = "configs/sncwgan_dtn.yaml"
cfg = OmegaConf.load(config_path)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion_mrae = Loss_MRAE()
criterion_rmse = Loss_RMSE()
criterion_psnr = Loss_PSNR()
criterion_psnrrgb = Loss_PSNR()
criterion_sam = Loss_SAM()
criterion_sid = Loss_SID()
criterion_fid = Loss_Fid().cuda()
criterion_ssim = Loss_SSIM().cuda()

class SpectralNormalizationCGAN(l.LightningModule):
    def __init__(self,
                 *,
                 modelconfig,
                 disconfig,
                 lossconfig, 
                 epochs,
                 learning_rate,
                 cond_key, 
                 num_warmup = 1,
                 n_critic = 2,
                 **kwargs
                 ):
        super().__init__()
        self.n_critic = n_critic
        self.cond_key = cond_key
        self.automatic_optimization = False
        self.end_epoch = epochs
        self.num_warmup = num_warmup
        self.lr = learning_rate
        self.global_step_manually = 0
        self._temp_epoch = 0
        self.iter_per_epoch = 760
        self.generator = instantiate_from_config(modelconfig)
        self.discriminator = instantiate_from_config(disconfig)
        self.loss = instantiate_from_config(lossconfig)
        gen_name = str(modelconfig.target).split('.')[-1]
        disc_name = str(disconfig.target).split('.')[-1]
        self.root = f'/work3/s212645/Spectral_Reconstruction/checkpoint/SwinGAN/{gen_name}_{disc_name}/'
        self.min_mrae = 1000
        self.freezeD = False
        self.finetune = False
        self.load_pth = False

    def cal_batch_size(self, device):
        torch.cuda.empty_cache()
        batch_size = 1
        input_data = Variable(torch.rand([1, 6, 128, 128]).to(device))
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_stats(device)["allocated_bytes.all.peak"]
        freememmory_now = torch.cuda.mem_get_info()[0]
        self.to(device)
        self.loss.to(device)
        self.eval()
        rec = self(input_data)
        self.discriminator(torch.concat([rec, input_data], dim=1))
        self.discriminator(torch.concat([rec, input_data], dim=1))
        memory_usage = torch.cuda.memory_stats(device)["allocated_bytes.all.peak"] - initial_memory
        print("GPU memory usage for batch size {} is {} GB".format(batch_size, memory_usage / 1024 ** 3))
        print(initial_memory / 1024 ** 3, 'GB')
        max_batch_size = freememmory_now // memory_usage
        print(int(max_batch_size))
        torch.cuda.empty_cache()
        return int(max_batch_size)
    
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
    
    def load_from_pth(self, path):
        sd = torch.load(path, map_location="cpu")
        self.generator.load_state_dict(sd['G'])
        self.discriminator.load_state_dict(sd['D'])
        self.pth_path = path
        self._temp_epoch = sd['epoch']
        self.best_mrae = sd['best_mrae']
        self.loss_min = sd['loss_min']
    
    def on_train_start(self) -> None:
        self.iter_per_epoch = len(self.trainer.train_dataloader)
        if not self.finetune:
            self.set_current_epoch(self._temp_epoch) # This is being loaded from the model
            total_batch_idx = self.current_epoch * len(self.trainer.train_dataloader)
            global_step = total_batch_idx
            self.set_global_step(global_step)
        self.global_step_manually = self.current_epoch * self.iter_per_epoch
        self.modify_iter = self.global_step_manually - self.global_step
        print(self.global_step_manually)

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

    def get_last_layer(self):
        return self.generator.mapping.weight

    def forward(self, input):
        reconstructions = self.generator(input)
        return reconstructions

    def training_step(self, batch, batch_idx):
        opt_g, opt_disc = self.optimizers()
        sch_g, sch_d = self.lr_schedulers()
        cond = batch[self.cond_key]
        labels = batch['label']
        cond = Variable(cond)
        labels = Variable(labels)
        rec = self(cond)

        if batch_idx % self.n_critic == 0 and not self.freezeD:
            # discriminator
            self.toggle_optimizer(opt_disc)
            discloss, log_dict_disc = self.loss(self.discriminator, rec, labels, cond, self.trainer.current_epoch, mode = 'dics', last_layer = self.get_last_layer())
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            self.manual_backward(discloss)
            opt_disc.step()
            opt_disc.zero_grad()
            self.untoggle_optimizer(opt_disc)

        # generator
        loss_mrae = criterion_mrae(rec, labels)
        # self.losses_mrae_train.update(loss_mrae.data)
        self.toggle_optimizer(opt_g)
        g_loss, log_dict_g = self.loss(self.discriminator, rec, labels, cond, self.trainer.current_epoch, mode = 'gen', last_layer = self.get_last_layer())
        # self.log("train/mrae_avg", self.losses_mrae_train.avg, prog_bar=True, logger=True, on_step=True, on_epoch=False, sync_dist=True)
        self.log("train/mrae", loss_mrae, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("lr", opt_g.param_groups[0]['lr'], prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_g, prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.manual_backward(g_loss)
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)

        self.global_step_manually += 1
        sch_g.step_update(self.global_step_manually)
        sch_d.step_update(self.global_step_manually)
    
    def on_train_epoch_start(self):
        self.losses_mrae_train = AverageMeter()
    
    def on_validation_epoch_start(self):
        self.losses_mrae = AverageMeter()
        self.losses_rmse = AverageMeter()
        self.losses_psnr = AverageMeter()
        self.losses_sam = AverageMeter()

    def validation_step(self, batch, batch_idx):
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
        self.log_dict({'mrae': loss_mrae, 'rmse': loss_rmse, 'psnr': loss_psnr, 'sam': loss_sam}, sync_dist=True)
        self.losses_mrae.update(loss_mrae.data)
        self.losses_rmse.update(loss_rmse.data)
        self.losses_psnr.update(loss_psnr.data)
        self.losses_sam.update(loss_sam.data)
        return self.log_dict
    
    def on_validation_epoch_end(self):
        if self.min_mrae > self.losses_mrae.avg:
            self.min_mrae = self.losses_mrae.avg
        print(f'validation: MRAE: {self.losses_mrae.avg}/{self.min_mrae}, RMSE: {self.losses_rmse.avg}, PSNR: {self.losses_psnr.avg}, SAM: {self.losses_sam.avg}.')
        self.log_dict({'val/mrae_avg': self.losses_mrae.avg, 'val/rmse_avg': self.losses_rmse.avg, 'val/psnr_avg': self.losses_psnr.avg, 'val/sam_avg': self.losses_sam.avg}, sync_dist=True)
    
    def configure_optimizers(self):
        self.trainer.fit_loop.setup_data()
        self.iter_per_epoch = len(self.trainer.train_dataloader)
        iter_per_epoch = self.iter_per_epoch
        lr = self.lr
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.9), eps=1e-14)
        opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.9), eps=1e-12)
        warmup_lr_init = 1e-3
        if self.finetune:
            sch_g = CosineLRScheduler(opt_g,t_initial=self.end_epoch * iter_per_epoch,
                                                cycle_mul = 1,cycle_decay = 1,lr_min=1e-6,
                                                warmup_lr_init=lr,warmup_t=0,
                                                cycle_limit=1,t_in_epochs=False)
            sch_disc = CosineLRScheduler(opt_disc,t_initial=self.end_epoch * iter_per_epoch,
                                                cycle_mul = 1,cycle_decay = 1,lr_min=1e-6,
                                                warmup_lr_init=lr,warmup_t=0,
                                                cycle_limit=1,t_in_epochs=False)
        else:
            sch_g = CosineLRScheduler(opt_g,t_initial=(self.end_epoch - self.num_warmup) * iter_per_epoch,
                                                cycle_mul = 1,cycle_decay = 1,lr_min=1e-6,
                                                warmup_lr_init=warmup_lr_init,warmup_t=self.num_warmup * iter_per_epoch,
                                                cycle_limit=1,t_in_epochs=False)
            sch_disc = CosineLRScheduler(opt_disc,t_initial=(self.end_epoch - self.num_warmup) * iter_per_epoch,
                                                cycle_mul = 1,cycle_decay = 1,lr_min=1e-6,
                                                warmup_lr_init=warmup_lr_init,warmup_t=self.num_warmup * iter_per_epoch,
                                                cycle_limit=1,t_in_epochs=False)
        if self.load_pth:
            try:
                sd = torch.load(self.pth_path, map_location="cpu")
                opt_g.load_state_dict(sd['optimG'])
                opt_disc.load_state_dict(sd['optimD'])
                print('optimizer loaded!!!!!!!!!!!!!!!!!!!')
            except Exception as ex:
                print(ex)
        return({"optimizer": opt_g, "lr_scheduler": sch_g},
                {"optimizer": opt_disc, "lr_scheduler": sch_disc},
                )