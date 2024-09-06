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
from timm.scheduler.cosine_lr import CosineLRScheduler

criterion_mrae = Loss_MRAE()
criterion_rmse = Loss_RMSE()
criterion_psnr = Loss_PSNR()
criterion_psnrrgb = Loss_PSNR()
criterion_sam = Loss_SAM()
criterion_sid = Loss_SID()
criterion_fid = Loss_Fid().cuda()
criterion_ssim = Loss_SSIM().cuda()
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
        self.automatic_optimization = False
        self.min_window = 128

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
        features = None
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior, features = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z, features)
        return dec

    def get_input(self, batch, k):
        x = batch[k]
        return x

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
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

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
    def __init__(self, ddconfig, condconfig, lossconfig, embed_dim, learning_rate, cond_ckpt_path, ckpt_path=None, ignore_keys=[], image_key="image", cond_key='cond', colorize_nlabels=None, monitor=None):
        super().__init__(ddconfig, lossconfig, embed_dim, ckpt_path, ignore_keys, image_key, colorize_nlabels, monitor, learning_rate)
        self.decoder = FirstStageDecoder(**ddconfig)
        self.cond_key = cond_key
        self.cond_encoder = load_lightning2torch(cond_ckpt_path, CondEncoder(**condconfig.params.ddconfig), name='encoder')
        self.cond_quant_conv = load_lightning2torch(cond_ckpt_path, torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1), name='quant_conv')

        self.automatic_optimization = False
        self.post_quant_conv = torch.nn.Conv2d(embed_dim*2, ddconfig["z_channels"]*2, 1)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def decode(self, z, condfeatures, h):
        if isinstance(h, DiagonalGaussianDistribution):
            h = h.mode()
        z = self.post_quant_conv(torch.concat([z, h], dim=1))
        dec = self.decoder(z, condfeatures)
        return dec

    def cond_encode(self, x):
        h, condfeatures = self.cond_encoder(x)
        moments = self.cond_quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior, condfeatures


    def forward(self, input, cond, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        h, condfeatures = self.cond_encode(cond)
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
        # output = self.sample(batch)
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
        h, condfeatures = self.cond_encode(cond)
        z = torch.randn_like(h.mode()).to(device)
        # h = self.cond_model.post_quant_conv(h.mode())
        z = self.post_quant_conv(torch.concat([z, h.mode()], dim=1))
        dec = self.decoder(z, condfeatures)
        return dec

class PerceptualVAE(l.LightningModule):
    def __init__(self,
                 *,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 epochs = 100,
                 num_warmup = 1,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="label",
                 colorize_nlabels=None,
                 monitor=None,
                 learning_rate = 4e-4,
                 **kwargs,
                 ):
        super().__init__()
        self.automatic_optimization = False
        self.image_key = image_key
        self.learning_rate = learning_rate
        self.end_epoch = epochs
        self.num_warmup = num_warmup
        self.min_mrae = 1000
        self.iter_per_epoch = 760
        self._temp_epoch = 0
        self._optimizer_states = None
        try:
            self.encoder = instantiate_from_config(ddconfig.encoder)
            self.decoder = instantiate_from_config(ddconfig.decoder)
        except Exception as ex:
            print(ex)
            # encoderconfig = OmegaConf.create()
            # decoderconfig = OmegaConf.create()
            # encoderconfig['target'] = ddconfig.encoder
            # encoderconfig['params'] = ddconfig
            # decoderconfig['target'] = ddconfig.decoder
            # decoderconfig['params'] = ddconfig
            # self.encoder = instantiate_from_config(encoderconfig)
            # self.decoder = instantiate_from_config(decoderconfig)
            self.encoder = DualTransformerEncoder(**ddconfig)
            self.decoder = DualTransformerDecoder(**ddconfig)
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

    def cal_batch_size(self, device):
        torch.cuda.empty_cache()
        batch_size = 1
        input_data = Variable(torch.rand([1, 31, 128, 128]).to(device))
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_stats(device)["allocated_bytes.all.peak"]
        freememmory_now = torch.cuda.mem_get_info()[0]
        self.to(device)
        self.loss.to(device)
        self.eval()
        reconstructions, posterior = self(input_data)
        self.loss.discriminator(reconstructions)
        self.loss.discriminator(reconstructions)
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
        self._temp_epoch = torch.load(path, map_location="cpu")['epoch']
        print(f"Restored from {path}")
        self.init_optim_ckpt(path)

    def init_optim_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["optimizer_states"]
        self._optimizer_states = sd
        print(f"Restored optimizer from {path}")

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
        self.trainer.fit_loop.setup_data()
        self.iter_per_epoch = len(self.trainer.train_dataloader)
        iter_per_epoch = self.iter_per_epoch
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        warmup_lr_init = lr
        sch_ae = CosineLRScheduler(opt_ae,t_initial=(self.end_epoch - self.num_warmup) * iter_per_epoch,
                                            cycle_mul = 1,cycle_decay = 1,lr_min=1e-6,
                                            warmup_lr_init=warmup_lr_init,warmup_t=self.num_warmup * iter_per_epoch,
                                            cycle_limit=1,t_in_epochs=False)
        sch_disc = CosineLRScheduler(opt_disc,t_initial=(self.end_epoch - self.num_warmup) * iter_per_epoch,
                                            cycle_mul = 1,cycle_decay = 1,lr_min=1e-6,
                                            warmup_lr_init=warmup_lr_init,warmup_t=self.num_warmup * iter_per_epoch,
                                            cycle_limit=1,t_in_epochs=False)
        if self._optimizer_states is not None:
            try:
                opt_ae.load_state_dict(self._optimizer_states[0])
                for group in opt_ae.param_groups:
                    group['lr'] = lr
                opt_disc.load_state_dict(self._optimizer_states[1])
                for group in opt_disc.param_groups:
                    group['lr'] = lr
            except:
                pass
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

        # train the discriminator
        self.toggle_optimizer(opt_disc)
        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

        self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False, sync_dist=True)
        self.manual_backward(discloss)
        opt_disc.step()
        opt_disc.zero_grad()
        self.untoggle_optimizer(opt_disc)
        
        # train the generator
        self.toggle_optimizer(opt_g)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        loss_rmse = criterion_rmse(reconstructions, inputs)
        aeloss += loss_rmse
        self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("recloss", log_dict_ae['train/rec_loss'], prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("lr", opt_g.param_groups[0]['lr'], prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        # self.log("step", self.global_step, prog_bar=True, logger=True, on_step=True, on_epoch=False, sync_dist=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False, sync_dist=True)
        self.manual_backward(aeloss)
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)
        
        self.iterations += 1
        sch_g.step_update(self.iterations)
        sch_d.step_update(self.iterations)
        
    def validation_step(self, batch, batch_idx):
        labels = batch[self.image_key]
        labels = Variable(labels)
        output, posterior = self(labels)
        loss_mrae = self.loss.l1_loss(output, labels).detach()
        loss_rmse = criterion_rmse(output, labels).detach()
        loss_psnr = criterion_psnr(output, labels).detach()
        loss_sam = criterion_sam(output, labels).detach()
        criterion_sam.reset()
        criterion_psnr.reset()
        self.log_dict({'val/mrae': loss_mrae, 'val/psnr': loss_psnr}, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict({'val/rmse': loss_rmse, 'val/sam': loss_sam}, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        return self.log_dict

    def test_step(self, batch, batch_idx):
        labels = batch[self.image_key]
        labels = Variable(labels)
        output, posterior = self(labels)
        loss_mrae = self.loss.l1_loss(output, labels).detach()
        loss_rmse = criterion_rmse(output, labels).detach()
        loss_psnr = criterion_psnr(output, labels).detach()
        loss_sam = criterion_sam(output, labels).detach()
        criterion_sam.reset()
        criterion_psnr.reset()
        self.log_dict({'test/mrae': loss_mrae, 'test/psnr': loss_psnr}, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict({'test/rmse': loss_rmse, 'test/sam': loss_sam}, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
    
    def on_train_start(self):
        self.iter_per_epoch = len(self.trainer.train_dataloader)
        self.set_current_epoch(self._temp_epoch) # This is being loaded from the model
        total_batch_idx = self.current_epoch * len(self.trainer.train_dataloader)
        global_step = total_batch_idx
        self.set_global_step(global_step)
        self.iterations = self.global_step
        print(self.current_epoch, self.global_step)

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

class PerceptualVAESkipConnect(PerceptualVAE):
    def __init__(self, *, ddconfig, lossconfig, embed_dim, cond_key = None, condconfig = None, epochs=100, num_warmup=1, ckpt_path=None, ignore_keys=[], image_key="label", colorize_nlabels=None, monitor=None, learning_rate=0.0004, **kwargs):
        super().__init__(ddconfig=ddconfig, lossconfig=lossconfig, embed_dim=embed_dim, epochs=epochs, num_warmup=num_warmup, ckpt_path=ckpt_path, ignore_keys=ignore_keys, image_key=image_key, colorize_nlabels=colorize_nlabels, monitor=monitor, learning_rate=learning_rate, **kwargs)
        # self.decoder = DualTransformerDecoderCond(**ddconfig)
        # self.decoder = instantiate_from_config(ddconfig)
        if condconfig is not None:
            assert cond_key is not None, 'cond key is None'
            self.cond_key = cond_key
            self.skip_connect = True
            self.cond_encoder = instantiate_from_config(condconfig)
            self.cond_encoder.eval()
        else:
            self.cond_key = None
            self.skip_connect = False
            self.cond_encoder = None
    
    def decode(self, z, shallow_features = None):
        z = self.post_quant_conv(z)
        dec = self.decoder(z, shallow_features)
        return dec

    def forward(self, input, cond = None, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        if self.skip_connect:
            assert cond is not None, 'cond is needed for shallow connection'
            shallow_features = self.cond_encoder.get_features(cond)
            dec = self.decode(z, shallow_features)
        else:
            dec = self.decode(z)
        return dec, posterior

    def training_step(self, batch, batch_idx):
        opt_g, opt_disc = self.optimizers()
        sch_g, sch_d = self.lr_schedulers()
        inputs = self.get_input(batch, self.image_key)
        if self.skip_connect:
            conds = self.get_input(batch, self.cond_key)
            reconstructions, posterior = self(inputs, conds)
        else:
            reconstructions, posterior = self(inputs)

        # train the discriminator
        self.toggle_optimizer(opt_disc)
        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

        self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False, sync_dist=True)
        self.manual_backward(discloss)
        opt_disc.step()
        opt_disc.zero_grad()
        self.untoggle_optimizer(opt_disc)

        # train the generator
        self.toggle_optimizer(opt_g)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        loss_rmse = criterion_rmse(reconstructions, inputs)
        aeloss += loss_rmse
        self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("recloss", log_dict_ae['train/rec_loss'], prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        # self.log("step", self.global_step, prog_bar=True, logger=True, on_step=True, on_epoch=False, sync_dist=True)
        self.log("lr", opt_g.param_groups[0]['lr'], prog_bar=True, logger=True, on_step=True, on_epoch=False, sync_dist=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False, sync_dist=True)
        self.manual_backward(aeloss)
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)
        
        self.iterations += 1
        sch_g.step_update(self.iterations)
        sch_d.step_update(self.iterations)
        
    def validation_step(self, batch, batch_idx):
        labels = batch[self.image_key]
        labels = Variable(labels)
        if self.skip_connect:
            conds = self.get_input(batch, self.cond_key)
            output, posterior = self(labels, conds)
        else:
            output, posterior = self(labels)
        loss_mrae = self.loss.l1_loss(output, labels).detach()
        loss_rmse = criterion_rmse(output, labels).detach()
        loss_psnr = criterion_psnr(output, labels).detach()
        loss_sam = criterion_sam(output, labels).detach()
        criterion_sam.reset()
        criterion_psnr.reset()
        self.log_dict({'val/mrae': loss_mrae, 'val/psnr': loss_psnr}, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict({'val/rmse': loss_rmse, 'val/sam': loss_sam}, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        return self.log_dict
    
    def test_step(self, batch, batch_idx):
        labels = batch[self.image_key]
        labels = Variable(labels)
        if self.skip_connect:
            conds = self.get_input(batch, self.cond_key)
            output, posterior = self(labels, conds)
        else:
            output, posterior = self(labels)
        loss_mrae = self.loss.l1_loss(output, labels).detach()
        loss_rmse = criterion_rmse(output, labels).detach()
        loss_psnr = criterion_psnr(output, labels).detach()
        loss_sam = criterion_sam(output, labels).detach()
        criterion_sam.reset()
        criterion_psnr.reset()
        self.log_dict({'test/mrae': loss_mrae, 'test/psnr': loss_psnr}, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict({'test/rmse': loss_rmse, 'test/sam': loss_sam}, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
    