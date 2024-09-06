import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from tqdm.autonotebook import tqdm
import numpy as np

from ldm.util import default
from ldm.modules.diffusionmodules.util import extract_into_tensor as extract
from ldm.modules.diffusionmodules.openaimodel import UNetModel

import torch
import torch.nn as nn
import numpy as np
import lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
from torchvision.utils import make_grid
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from ldm.modules.ema import LitEma
from ldm.modules.distributions.distributions import normal_kl
from models.vae.networks import DiagonalGaussianDistribution
from ldm.models.autoencoder import VQModelInterface, IdentityFirstStage, AutoencoderKL
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from ldm.models.diffusion.ddim import DDIMSampler
from utils import *
from timm.scheduler.cosine_lr import CosineLRScheduler

criterion_mrae = Loss_MRAE()
criterion_rmse = Loss_RMSE()
criterion_psnr = Loss_PSNR()
criterion_psnrrgb = Loss_PSNR()
criterion_sam = Loss_SAM()
criterion_sid = Loss_SID()
criterion_fid = Loss_Fid().cuda()
criterion_ssim = Loss_SSIM().cuda()

class BrownianBridgeModel(pl.LightningModule):
    def __init__(self, * , 
                 unet_config,
                 learning_rate, 
                 timesteps=1000,
                 mt_type = 'linear',
                 max_var = 1.0,
                 eta = 1.0,
                 skip_sample = True, 
                 sample_type = 'linear',
                 sample_step = 200,
                 loss_type = 'l1',
                 objective = 'grad',
                 image_size=256,
                 channels=3,
                 first_stage_key="image",
                 cond_stage_key="ycrcb",
                 num_warmup = 0,
                 epochs = 500,
                 start_sample = 250,
                 warmup_lr = None,
                 **kwargs
                 ):
        super().__init__()
        assert start_sample < epochs, 'epoch must larger than start_sample'
        self.start_sample = start_sample
        self.automatic_optimization = False
        # model hyperparameters
        self.num_timesteps = timesteps
        self.mt_type = mt_type
        self.max_var = max_var
        self.eta = eta
        self.skip_sample = skip_sample
        self.sample_type = sample_type
        self.sample_step = sample_step
        self.steps = None
        self.register_schedule()
        self.end_epoch = epochs
        self.num_warmup = num_warmup
        self.learning_rate = learning_rate
        self._optimizer_states = None
        self._temp_epoch = 0
        self.sampling_logs_with_images = False
        self.warmup_lr = warmup_lr

        # loss and objective
        self.loss_type = loss_type
        self.objective = objective

        # UNet
        self.image_size = image_size
        self.channels = channels
        self.cond_stage_key = cond_stage_key
        self.first_stage_key = first_stage_key

        self.denoise_fn = instantiate_from_config(unet_config)

    def register_schedule(self):
        T = self.num_timesteps

        if self.mt_type == "linear":
            m_min, m_max = 0.001, 0.999
            m_t = np.linspace(m_min, m_max, T)
        elif self.mt_type == "sin":
            m_t = 1.0075 ** np.linspace(0, T, T)
            m_t = m_t / m_t[-1]
            m_t[-1] = 0.999
        else:
            raise NotImplementedError
        m_tminus = np.append(0, m_t[:-1])

        variance_t = 2. * (m_t - m_t ** 2) * self.max_var
        variance_tminus = np.append(0., variance_t[:-1])
        variance_t_tminus = variance_t - variance_tminus * ((1. - m_t) / (1. - m_tminus)) ** 2
        posterior_variance_t = variance_t_tminus * variance_tminus / variance_t

        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('m_t', to_torch(m_t))
        self.register_buffer('m_tminus', to_torch(m_tminus))
        self.register_buffer('variance_t', to_torch(variance_t))
        self.register_buffer('variance_tminus', to_torch(variance_tminus))
        self.register_buffer('variance_t_tminus', to_torch(variance_t_tminus))
        self.register_buffer('posterior_variance_t', to_torch(posterior_variance_t))

        if self.skip_sample:
            if self.sample_type == 'linear':
                midsteps = torch.arange(self.num_timesteps - 1, 1,
                                        step=-((self.num_timesteps - 1) / (self.sample_step - 2))).long()
                self.steps = torch.cat((midsteps, torch.Tensor([1, 0]).long()), dim=0)
            elif self.sample_type == 'cosine':
                steps = np.linspace(start=0, stop=self.num_timesteps, num=self.sample_step + 1)
                steps = (np.cos(steps / self.num_timesteps * np.pi) + 1.) / 2. * self.num_timesteps
                self.steps = torch.from_numpy(steps)
        else:
            self.steps = torch.arange(self.num_timesteps-1, -1, -1)

    def apply(self, weight_init):
        self.denoise_fn.apply(weight_init)
        return self

    def get_parameters(self):
        return self.denoise_fn.parameters()

    def forward(self, x, y, context=None):
        if self.cond_stage_key == "nocond":
            context = None
        else:
            context = y if context is None else context
        b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, y, context, t)

    def p_losses(self, x0, y, context, t, noise=None):
        """
        model loss
        :param x0: encoded x_ori, E(x_ori) = x0
        :param y: encoded y_ori, E(y_ori) = y
        :param y_ori: original source domain image
        :param t: timestep
        :param noise: Standard Gaussian Noise
        :return: loss
        """
        b, c, h, w = x0.shape
        noise = default(noise, lambda: torch.randn_like(x0))

        x_t, objective = self.q_sample(x0, y, t, noise)
        objective_recon = self.denoise_fn(x_t, timesteps=t, context=y)

        if self.loss_type == 'l1':
            recloss = (objective - objective_recon).abs().mean()
        elif self.loss_type == 'l2':
            recloss = F.mse_loss(objective, objective_recon)
        elif self.loss_type == 'huber':
            recloss = F.mse_loss(objective, objective_recon) + F.l1_loss(objective, objective_recon)
        else:
            raise NotImplementedError()

        x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon)
        log_dict = {
            "loss_simple": recloss,
            # "x0_recon": x0_recon
        }
        return recloss, log_dict

    def q_sample(self, x0, y, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x0))
        m_t = extract(self.m_t, t, x0.shape)
        var_t = extract(self.variance_t, t, x0.shape)
        sigma_t = torch.sqrt(var_t)

        if self.objective == 'grad':
            objective = m_t * (y - x0) + sigma_t * noise
        elif self.objective == 'noise':
            objective = noise
        elif self.objective == 'ysubx':
            objective = y - x0
        else:
            raise NotImplementedError()

        return (
            (1. - m_t) * x0 + m_t * y + sigma_t * noise,
            objective
        )

    def predict_x0_from_objective(self, x_t, y, t, objective_recon):
        if self.objective == 'grad':
            x0_recon = x_t - objective_recon
        elif self.objective == 'noise':
            m_t = extract(self.m_t, t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)
            sigma_t = torch.sqrt(var_t)
            x0_recon = (x_t - m_t * y - sigma_t * objective_recon) / (1. - m_t)
        elif self.objective == 'ysubx':
            x0_recon = y - objective_recon
        else:
            raise NotImplementedError
        return x0_recon

    @torch.no_grad()
    def q_sample_loop(self, x0, y):
        imgs = [x0]
        for i in tqdm(range(self.num_timesteps), desc='q sampling loop', total=self.num_timesteps):
            t = torch.full((y.shape[0],), i, device=x0.device, dtype=torch.long)
            img, _ = self.q_sample(x0, y, t)
            imgs.append(img)
        return imgs

    @torch.no_grad()
    def p_sample(self, x_t, y, context, i, clip_denoised=False):
        b, *_, device = *x_t.shape, x_t.device
        if self.steps[i] == 0:
            t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
            objective_recon = self.denoise_fn(x_t, timesteps=t, context=context)
            x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon)
            if clip_denoised:
                x0_recon.clamp_(-1., 1.)
            return x0_recon, x0_recon
        else:
            t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
            n_t = torch.full((x_t.shape[0],), self.steps[i+1], device=x_t.device, dtype=torch.long)

            objective_recon = self.denoise_fn(x_t, timesteps=t, context=context)
            x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon)
            if clip_denoised:
                x0_recon.clamp_(-1., 1.)

            m_t = extract(self.m_t, t, x_t.shape)
            m_nt = extract(self.m_t, n_t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)
            var_nt = extract(self.variance_t, n_t, x_t.shape)
            sigma2_t = (var_t - var_nt * (1. - m_t) ** 2 / (1. - m_nt) ** 2) * var_nt / var_t
            sigma_t = torch.sqrt(sigma2_t) * self.eta

            noise = torch.randn_like(x_t)
            x_tminus_mean = (1. - m_nt) * x0_recon + m_nt * y + torch.sqrt((var_nt - sigma2_t) / var_t) * \
                            (x_t - (1. - m_t) * x0_recon - m_t * y)

            return x_tminus_mean + sigma_t * noise, x0_recon

    @torch.no_grad()
    def p_sample_loop(self, y, context=None, clip_denoised=True, sample_mid_step=False):
        if self.cond_stage_key == "nocond":
            context = None
        else:
            context = y if context is None else context

        if sample_mid_step:
            imgs, one_step_imgs = [y], []
            for i in tqdm(range(len(self.steps)), desc=f'sampling loop time step', total=len(self.steps)):
                img, x0_recon = self.p_sample(x_t=imgs[-1], y=y, context=context, i=i, clip_denoised=clip_denoised)
                imgs.append(img)
                one_step_imgs.append(x0_recon)
            return imgs, one_step_imgs
        else:
            img = y
            for i in tqdm(range(len(self.steps)), desc=f'sampling loop time step', total=len(self.steps)):
                img, _ = self.p_sample(x_t=img, y=y, context=context, i=i, clip_denoised=clip_denoised)
            return img

    @torch.no_grad()
    def sample(self, y, context=None, clip_denoised=True, sample_mid_step=False):
        return self.p_sample_loop(y, context, clip_denoised, sample_mid_step)
    
    
    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        self._temp_epoch = torch.load(path, map_location="cpu")['epoch']
        print(self._temp_epoch)
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")
        self.init_optim_ckpt(path)

    def init_optim_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["optimizer_states"]
        self._optimizer_states = sd
        print(f"Restored optimizer from {path}")

    def get_input(self, batch, k):
        x = batch[k]
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    
    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        sch = self.lr_schedulers()
        self.toggle_optimizer(opt)
        x = self.get_input(batch, k=self.first_stage_key)
        y = self.get_input(batch, k=self.cond_stage_key)
        loss, loss_dict = self(x, y)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        self.manual_backward(loss)
        opt.step()
        opt.zero_grad()
        self.untoggle_optimizer(opt)
        sch.step_update(self.global_step)

        return loss

    def on_validation_epoch_start(self):
        self.losses_mrae = AverageMeter()
        self.losses_rmse = AverageMeter()
        self.losses_psnr = AverageMeter()
        self.losses_sam = AverageMeter()

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, k=self.first_stage_key)
        y = self.get_input(batch, k=self.cond_stage_key)
        _, loss_dict_no_ema = self(x, y)
        
        loss_dict_no_ema = {'val/' + key: loss_dict_no_ema[key] for key in loss_dict_no_ema}
        # with self.ema_scope():
        #     _, loss_dict_ema = self.shared_step(batch)
        #     loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        # self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.__monitor = loss_dict_no_ema['val/loss_simple']

        # if self.current_epoch % 5 == 0 and self.current_epoch > 0 and self.sampling_logs_with_images:
        if self.sampling_logs_with_images:
            log = self.log_images(batch, N = batch['label'].shape[0])
            xrec = log['samples']
            loss_mrae = criterion_mrae(xrec, batch[self.first_stage_key]).detach()
            loss_rmse = criterion_rmse(xrec, batch[self.first_stage_key]).detach()
            loss_psnr = criterion_psnr(xrec, batch[self.first_stage_key]).detach()
            loss_sam = criterion_sam(xrec, batch[self.first_stage_key]).detach()
            criterion_sam.reset()
            criterion_psnr.reset()
            self.log_dict({'mrae': loss_mrae, 'rmse': loss_rmse, 'psnr': loss_psnr, 'sam': loss_sam}, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.losses_mrae.update(loss_mrae.data)
            self.losses_rmse.update(loss_rmse.data)
            self.losses_psnr.update(loss_psnr.data)
            self.losses_sam.update(loss_sam.data)

    def on_validation_epoch_end(self):
        if self.sampling_logs_with_images:
            print(f'validation: MRAE: {self.losses_mrae.avg}, RMSE: {self.losses_rmse.avg}, PSNR: {self.losses_psnr.avg}, SAM: {self.losses_sam.avg}.')
            self.log_dict({'val/mrae_avg': self.losses_mrae.avg, 'val/rmse_avg': self.losses_rmse.avg, 'val/psnr_avg': self.losses_psnr.avg, 'val/sam_avg': self.losses_sam.avg})
        if self.current_epoch % 5 == 0 and self.current_epoch > self.start_sample:
        # and self.__monitor < 0.4:
            self.sampling_logs_with_images = True

    def on_test_epoch_start(self):
        self.losses_mrae = AverageMeter()
        self.losses_rmse = AverageMeter()
        self.losses_psnr = AverageMeter()
        self.losses_sam = AverageMeter()

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        x = self.get_input(batch, k=self.first_stage_key)
        y = self.get_input(batch, k=self.cond_stage_key)
        _, loss_dict_no_ema = self(x, y)
        loss_dict_no_ema = {'val/' + key: loss_dict_no_ema[key] for key in loss_dict_no_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        log = self.log_images(batch, N = batch['label'].shape[0])
        xrec = log['samples']
        loss_mrae = criterion_mrae(xrec, batch[self.first_stage_key]).detach()
        loss_rmse = criterion_rmse(xrec, batch[self.first_stage_key]).detach()
        loss_psnr = criterion_psnr(xrec, batch[self.first_stage_key]).detach()
        loss_sam = criterion_sam(xrec, batch[self.first_stage_key]).detach()
        criterion_sam.reset()
        criterion_psnr.reset()
        self.log_dict({'mrae': loss_mrae, 'rmse': loss_rmse, 'psnr': loss_psnr, 'sam': loss_sam}, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.losses_mrae.update(loss_mrae.data)
        self.losses_rmse.update(loss_rmse.data)
        self.losses_psnr.update(loss_psnr.data)
        self.losses_sam.update(loss_sam.data)

    def on_test_epoch_end(self):
        print(f'validation: MRAE: {self.losses_mrae.avg}, RMSE: {self.losses_rmse.avg}, PSNR: {self.losses_psnr.avg}, SAM: {self.losses_sam.avg}.')
        self.log_dict({'val/mrae_avg': self.losses_mrae.avg, 'val/rmse_avg': self.losses_rmse.avg, 'val/psnr_avg': self.losses_psnr.avg, 'val/sam_avg': self.losses_sam.avg})

    def configure_optimizers(self):
        self.trainer.fit_loop.setup_data()
        self.iter_per_epoch = len(self.trainer.train_dataloader)
        iter_per_epoch = self.iter_per_epoch
        lr = self.learning_rate
        params = list(self.denoise_fn.parameters())
        # if self.learn_logvar:
        #     params = params + [self.logvar]
        opt = torch.optim.AdamW(params, lr=lr)
        if self.warmup_lr is not None:
            warmup_lr_init = self.warmup_lr
        else:
            warmup_lr_init = lr * 10
        scheduler = CosineLRScheduler(opt,t_initial=(self.end_epoch - self.num_warmup) * iter_per_epoch,
                                            cycle_mul = 1,cycle_decay = 1,lr_min=1e-6,
                                            warmup_lr_init=warmup_lr_init,warmup_t=self.num_warmup * iter_per_epoch,
                                            cycle_limit=1,t_in_epochs=False)
        if self._optimizer_states is not None:
            try:
                opt.load_state_dict(self._optimizer_states[0])
                for group in opt.param_groups:
                    group['lr'] = lr
            except:
                pass
        return [opt], scheduler
    
    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs):
        log = dict()
        x = self.get_input(batch, self.first_stage_key)
        y = self.get_input(batch, self.cond_stage_key)
        log["inputs"] = x
        samples = self.sample(y)
        log["samples"] = samples
        return log

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
