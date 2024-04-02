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
from options import opt
import numpy as np
from models.vae.Base import BaseModel
from models.transformer.MST_Plus_Plus import *
import lightning as l
from models.vae.networks import *

criterion_mrae = Loss_MRAE()
criterion_rmse = Loss_RMSE()
criterion_psnr = Loss_PSNR()
criterion_psnrrgb = Loss_PSNR()
criterion_sam = Loss_SAM()
criterion_sid = Loss_SID()
criterion_fid = Loss_Fid().cuda()
criterion_ssim = Loss_SSIM().cuda()

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
if opt.multigpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AutoencoderKL(l.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="label",
                 colorize_nlabels=None,
                 monitor=None,
                 learning_rate = 4e-4,
                 ):
        super().__init__()
        self.image_key = image_key
        self.learning_rate = learning_rate
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

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

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class CondAutoencoderKL(AutoencoderKL):
    def __init__(self, ddconfig, lossconfig, embed_dim, learning_rate, ckpt_path=None, ignore_keys=[], image_key="image", colorize_nlabels=None, monitor=None):
        super().__init__(ddconfig, lossconfig, embed_dim, ckpt_path, ignore_keys, image_key, colorize_nlabels, monitor, learning_rate)
        self.encoder = CondEncoder(**ddconfig)
        self.automatic_optimization = False

    def encode(self, x):
        h, condfeatures = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior, condfeatures

    def forward(self, input, sample_posterior=True):
        posterior, condfeatures = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior, condfeatures

    def training_step(self, batch, batch_idx):
        opt_g, opt_disc = self.optimizers()
        inputs = batch[self.image_key]
        inputs = Variable(inputs)

        reconstructions, posterior, condfeatures = self(inputs)

        self.toggle_optimizer(opt_g)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        self.manual_backward(aeloss)
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)
        
            # train the discriminator
        self.toggle_optimizer(opt_disc)
        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

        self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
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
        labels = batch[self.image_key]
        labels = Variable(labels)
        output, posterior, condfeatures = self(labels)
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
        return self.log_dict
    
    def on_validation_epoch_end(self):
        print(f'validation: MRAE: {self.losses_mrae.avg}, RMSE: {self.losses_rmse.avg}, PSNR: {self.losses_psnr.avg}, SAM: {self.losses_sam.avg}.')
        self.log_dict({'val/mrae_avg': self.losses_mrae.avg, 'val/rmse_avg': self.losses_rmse.avg, 'val/psnr_avg': self.losses_psnr.avg, 'val/sam_avg': self.losses_sam.avg})
    

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class FirstStageAutoencoderKL(AutoencoderKL):
    def __init__(self, ddconfig, condconfig, lossconfig, embed_dim, learning_rate, ckpt_path=None, ignore_keys=[], image_key="image", cond_key='cond', colorize_nlabels=None, monitor=None):
        super().__init__(ddconfig, lossconfig, embed_dim, ckpt_path, ignore_keys, image_key, colorize_nlabels, monitor, learning_rate)
        self.decoder = FirstStageDecoder(**ddconfig)
        self.cond_key = cond_key
        self.cond_model = instantiate_from_config(condconfig)
        self.cond_model = self.cond_model.eval()
        self.cond_model.train = disabled_train
        self.automatic_optimization = False
        self.post_quant_conv = torch.nn.Conv2d(embed_dim*2, ddconfig["z_channels"]*2, 1)

    def decode(self, z, condfeatures, h):
        z = self.post_quant_conv(torch.concat([z, h.mode()], dim=1))
        # h = self.cond_model.post_quant_conv(h.mode())
        dec = self.decoder(z, condfeatures)
        return dec

    def forward(self, input, cond, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        h, condfeatures = self.cond_model.encode(cond)
        dec = self.decode(z, condfeatures, h)
        return dec, posterior
    
    def training_step(self, batch, batch_idx):
        opt_g, opt_disc = self.optimizers()
        inputs = self.get_input(batch, self.image_key)
        cond = self.get_input(batch, self.cond_key)
        reconstructions, posterior = self(inputs, cond)

        self.toggle_optimizer(opt_g)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), cond=cond, split="train")
        self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        self.manual_backward(aeloss)
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)
        
            # train the discriminator
        self.toggle_optimizer(opt_disc)
        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), cond=cond, split="train")

        self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
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
        labels = batch[self.image_key]
        labels = Variable(labels)
        cond = batch[self.cond_key]
        cond = Variable(cond)
        output, posterior = self(labels, cond)
        output = self.sample(batch)
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
        return self.log_dict
    
    def on_validation_epoch_end(self):
        print(f'validation: MRAE: {self.losses_mrae.avg}, RMSE: {self.losses_rmse.avg}, PSNR: {self.losses_psnr.avg}, SAM: {self.losses_sam.avg}.')
        self.log_dict({'val/mrae_avg': self.losses_mrae.avg, 'val/rmse_avg': self.losses_rmse.avg, 'val/psnr_avg': self.losses_psnr.avg, 'val/sam_avg': self.losses_sam.avg})
    
    def sample(self, batch):
        cond = batch[self.cond_key]
        cond = Variable(cond)
        h, condfeatures = self.cond_model.encode(cond)
        z = torch.randn_like(h.mode()).to(device)
        # h = self.cond_model.post_quant_conv(h.mode())
        z = self.post_quant_conv(torch.concat([z, h.mode()], dim=1))
        dec = self.decoder(z, condfeatures)
        return dec

class PerceptualVAE(l.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="label",
                 colorize_nlabels=None,
                 monitor=None,
                 learning_rate = 4e-4,
                 ):
        super().__init__()
        self.automatic_optimization = False
        self.image_key = image_key
        self.learning_rate = learning_rate
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

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

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        return x
    
    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        sch_ae = torch.optim.lr_scheduler.CosineAnnealingLR(opt_ae, 1000, eta_min=1e-6)
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        sch_disc = torch.optim.lr_scheduler.CosineAnnealingLR(opt_disc, 1000, eta_min=1e-6)
        # return [opt_ae, opt_disc], []
        return({"optimizer": opt_ae, "lr_scheduler": sch_ae},
                {"optimizer": opt_disc, "lr_scheduler": sch_disc},
                )

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x
    
    def training_step(self, batch, batch_idx):
        opt_g, opt_disc = self.optimizers()
        sch_g, sch_d = self.lr_schedulers()
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)

        self.toggle_optimizer(opt_g)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        self.manual_backward(aeloss)
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)
        
            # train the discriminator
        self.toggle_optimizer(opt_disc)
        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

        self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        self.manual_backward(discloss)
        opt_disc.step()
        opt_disc.zero_grad()
        self.untoggle_optimizer(opt_disc)

        if self.trainer.is_last_batch and (self.trainer.current_epoch + 1) % 1 == 0:
            sch_g.step()
            sch_d.step()
        
    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.losses_mrae = AverageMeter()
            self.losses_rmse = AverageMeter()
            self.losses_psnr = AverageMeter()
            self.losses_sam = AverageMeter()
        labels = batch[self.image_key]
        labels = Variable(labels)
        output, posterior = self(labels)
        # output = self(batch)
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
        return self.log_dict
    
    def on_validation_epoch_end(self):
        print(f'validation: MRAE: {self.losses_mrae.avg}, RMSE: {self.losses_rmse.avg}, PSNR: {self.losses_psnr.avg}, SAM: {self.losses_sam.avg}.')
        self.log_dict({'val/mrae_avg': self.losses_mrae.avg, 'val/rmse_avg': self.losses_rmse.avg, 'val/psnr_avg': self.losses_psnr.avg, 'val/sam_avg': self.losses_sam.avg})
    