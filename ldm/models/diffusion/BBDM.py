"""
wild mixture of
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
-- merci
"""

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
from ldm.modules.diffusionmodules.openaimodel import UNetModel

criterion_mrae = Loss_MRAE()
criterion_rmse = Loss_RMSE()
criterion_psnr = Loss_PSNR()
criterion_psnrrgb = Loss_PSNR()
criterion_sam = Loss_SAM()
criterion_sid = Loss_SID()
criterion_fid = Loss_Fid().cuda()
criterion_ssim = Loss_SSIM().cuda()

__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def uniform_on_device(r1, r2, shape, device):
    return (r1 - r2) * torch.rand(*shape, device=device) + r2


class BrownianBridgeModel(pl.LightningModule):
    def __init__(self,*args, 
                 unet_config,
                 timesteps=1000,
                 beta_schedule="linear",
                 loss_type="l2",
                #  ckpt_path=None,
                #  ignore_keys=[],
                #  load_only_unet=False,
                #  monitor="val/loss",
                #  use_ema=True,
                 first_stage_key="image",
                 cond_stage_key="ycrcb",
                 image_size=256,
                 channels=3,
                 log_every_t=200,
                 epochs = 100,
                 learning_rate,
                #  clip_denoised=True,
                #  linear_start=1e-4,
                #  linear_end=2e-2,
                #  cosine_s=8e-3,
                #  given_betas=None,
                #  original_elbo_weight=0.,
                #  v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                #  l_simple_weight=1.,
                #  conditioning_key=None,
                #  parameterization="eps",  # all assuming fixed variance schedules
                 scheduler_config=None,
                #  use_positional_encodings=False,
                #  learn_logvar=False,
                #  logvar_init=0.,
                 num_warmup = 0,
                 **kwargs):
        super().__init__()
        self.model_config = unet_config
        self.log_every_t = log_every_t
        self.end_epoch = epochs
        self.num_warmup = num_warmup
        self.num_timesteps = timesteps
        self.mt_type = beta_schedule
        self.max_var = 1.0
        self.eta = 1.0
        self.skip_sample = True
        self.sample_type = beta_schedule
        self.sample_step = log_every_t
        self.steps = None
        self.register_schedule()
        self.use_scheduler = True
        self.scheduler_config = scheduler_config
        self._temp_epoch = 0
        self._optimizer_states = None
        self.automatic_optimization = False
        self.learning_rate = learning_rate

        # loss and objective
        self.loss_type = loss_type
        self.objective = 'grad'

        # UNet
        self.image_size = image_size
        self.channels = channels
        self.condition_key = cond_stage_key
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
        # if self.condition_key == "nocond":
        #     context = None
        # else:
        #     context = y if context is None else context
        context = y
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
        objective_recon = self.denoise_fn(x_t, timesteps=t, context=context)

        if self.loss_type == 'l1':
            recloss = (objective - objective_recon).abs().mean()
        elif self.loss_type == 'l2':
            recloss = F.mse_loss(objective, objective_recon)
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
        m_t = extract_into_tensor(self.m_t, t, x0.shape)
        var_t = extract_into_tensor(self.variance_t, t, x0.shape)
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
            m_t = extract_into_tensor(self.m_t, t, x_t.shape)
            var_t = extract_into_tensor(self.variance_t, t, x_t.shape)
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

            m_t = extract_into_tensor(self.m_t, t, x_t.shape)
            m_nt = extract_into_tensor(self.m_t, n_t, x_t.shape)
            var_t = extract_into_tensor(self.variance_t, t, x_t.shape)
            var_nt = extract_into_tensor(self.variance_t, n_t, x_t.shape)
            sigma2_t = (var_t - var_nt * (1. - m_t) ** 2 / (1. - m_nt) ** 2) * var_nt / var_t
            sigma_t = torch.sqrt(sigma2_t) * self.eta

            noise = torch.randn_like(x_t)
            x_tminus_mean = (1. - m_nt) * x0_recon + m_nt * y + torch.sqrt((var_nt - sigma2_t) / var_t) * \
                            (x_t - (1. - m_t) * x0_recon - m_t * y)

            return x_tminus_mean + sigma_t * noise, x0_recon

    @torch.no_grad()
    def p_sample_loop(self, y, context=None, clip_denoised=True, sample_mid_step=False):
        if self.condition_key == "nocond":
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

    def shared_step(self, batch):
        x = self.get_input(batch, self.first_stage_key)
        c = self.get_input(batch, self.cond_stage_key)
        loss, loss_dict = self(x, c)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        sch = self.lr_schedulers()
        self.toggle_optimizer(opt)
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
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
        _, loss_dict_no_ema = self.shared_step(batch)
        
        loss_dict_no_ema = {'val/' + key: loss_dict_no_ema[key] for key in loss_dict_no_ema}
        # with self.ema_scope():
        #     _, loss_dict_ema = self.shared_step(batch)
        #     loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        # self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.__monitor = loss_dict_no_ema['val/loss_simple']

        if self.current_epoch % 5 == 0 and self.current_epoch > 0 and self.__monitor < 0.4:
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
        if self.current_epoch % 5 == 0 and self.current_epoch > 0 and self.__monitor < 0.01:
            print(f'validation: MRAE: {self.losses_mrae.avg}, RMSE: {self.losses_rmse.avg}, PSNR: {self.losses_psnr.avg}, SAM: {self.losses_sam.avg}.')
            self.log_dict({'val/mrae_avg': self.losses_mrae.avg, 'val/rmse_avg': self.losses_rmse.avg, 'val/psnr_avg': self.losses_psnr.avg, 'val/sam_avg': self.losses_sam.avg})
    
    # def on_train_batch_end(self, *args, **kwargs):
    #     if self.use_ema:
    #         self.model_ema(self.model)

    def _get_rows_from_list(self, samples):
        n_imgs_per_row = len(samples)
        denoise_grid = rearrange(samples, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs):
        log = dict()
        x = self.get_input(batch, self.first_stage_key)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        x = x.to(self.device)[:N]
        log["inputs"] = x

        # get diffusion row
        diffusion_row = list()
        x_start = x[:n_row]

        for t in range(self.num_timesteps):
            if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                t = t.to(self.device).long()
                noise = torch.randn_like(x_start)
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                diffusion_row.append(x_noisy)

        log["diffusion_row"] = self._get_rows_from_list(diffusion_row)

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, denoise_row = self.sample(batch_size=N, return_intermediates=True)

            log["samples"] = samples
            log["denoise_row"] = self._get_rows_from_list(denoise_row)

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self):
        self.trainer.fit_loop.setup_data()
        self.iter_per_epoch = len(self.trainer.train_dataloader)
        iter_per_epoch = self.iter_per_epoch
        lr = self.learning_rate
        params = list(self.denoise_fn.parameters())
        # if self.learn_logvar:
        #     params = params + [self.logvar]
        opt = torch.optim.AdamW(params, lr=lr)
        warmup_lr_init = lr
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

class LatentDiffusion(BrownianBridgeModel):
    """main class"""
    def __init__(self,*args, 
                 unet_config,
                 timesteps=1000,
                 beta_schedule="linear",
                 loss_type="l2",
                 first_stage_key="label",
                 cond_stage_key="ycrcb",
                 image_size=256,
                 channels=3,
                 log_every_t=100,
                 epochs = 100,
                 learning_rate,
                 scheduler_config=None,
                 first_stage_config,
                 cond_stage_config,
                 num_timesteps_cond=None,
                 cond_stage_trainable=False,
                 concat_mode=True,
                 cond_stage_forward=None,
                 conditioning_key=None,
                 scale_factor=1.0,
                 scale_by_std=False,
                 use_features = False, 
                 num_warmups = 0,
                 **kwargs):
        super().__init__(
                 unet_config=unet_config,
                 learning_rate=learning_rate,
                 timesteps=timesteps,
                 beta_schedule=beta_schedule,
                 loss_type=loss_type,
                 first_stage_key=first_stage_key,
                 cond_stage_key=cond_stage_key,
                 image_size=image_size,
                 channels=channels,
                 log_every_t=log_every_t,
                 epochs=epochs,
                 scheduler_config=scheduler_config,
                 num_warmups=num_warmups)
        self.use_features = use_features
        self.learning_rate = learning_rate
        # self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        # assert self.num_timesteps_cond <= kwargs['timesteps']
        # for backwards compatibility after implementation of DiffusionWrapper
        if conditioning_key is None:
            conditioning_key = 'concat' if concat_mode else 'crossattn'
        if cond_stage_config == '__is_unconditional__':
            conditioning_key = None
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))
        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None  
        self._temp_epoch = 0

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True

    def make_cond_schedule(self, ):
        self.cond_ids = torch.full(size=(self.num_timesteps,), fill_value=self.num_timesteps - 1, dtype=torch.long)
        ids = torch.round(torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)).long()
        self.cond_ids[:self.num_timesteps_cond] = ids

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx):
        # only for very first batch
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0 and not self.restarted_from_ckpt:
            assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
            # set rescale weight to 1./std of encodings
            print("### USING STD-RESCALING ###")
            x = super().get_input(batch, self.first_stage_key)
            x = x.to(self.device)
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer('scale_factor', 1. / z.flatten().std())
            print(f"setting self.scale_factor to {self.scale_factor}")
            print("### USING STD-RESCALING ###")

    def get_ema_net(self):
        return self

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        del self.first_stage_model.loss
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            if config == "__is_first_stage__":
                print("Using first stage also as cond stage.")
                self.cond_stage_model = self.first_stage_model
            elif config == "__is_unconditional__":
                print(f"Training {self.__class__.__name__} as an unconditional model.")
                self.cond_stage_model = None
                # self.be_unconditional = True
            else:
                model = instantiate_from_config(config)
                self.cond_stage_model = model.eval()
                self.cond_stage_model.train = disabled_train
                for param in self.cond_stage_model.parameters():
                    param.requires_grad = False
        else:
            assert config != '__is_first_stage__'
            assert config != '__is_unconditional__'
            model = instantiate_from_config(config)
            self.cond_stage_model = model
        del self.cond_stage_model.loss
        # del self.cond_stage_model.decoder

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.mode()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                c = self.cond_stage_model.encode(c)
                # c, features = self.first_stage_model.cond_encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    def get_ema_net(self):
        return self

    # def get_parameters(self):
    #     if self.condition_key == 'SpatialRescaler':
    #         print("get parameters to optimize: SpatialRescaler, UNet")
    #         params = itertools.chain(self.denoise_fn.parameters(), self.cond_stage_model.parameters())
    #     else:
    #         print("get parameters to optimize: UNet")
    #         params = self.denoise_fn.parameters()
    #     return params

    def apply(self, weights_init):
        super().apply(weights_init)
        if self.cond_stage_model is not None:
            self.cond_stage_model.apply(weights_init)
        return self

    # def forward(self, x, x_cond, context=None):
    #     with torch.no_grad():
    #         x_latent = self.encode(x, cond=False)
    #         x_cond_latent = self.encode(x_cond, cond=True)
    #     context = self.get_cond_stage_context(x_cond)
    #     return super().forward(x_latent.detach(), x_cond_latent.detach(), context)

    def get_cond_stage_context(self, x_cond):
        if self.cond_stage_model is not None:
            context = self.cond_stage_model(x_cond)
            if self.condition_key == 'first_stage':
                context = context.detach()
        else:
            context = None
        return context

    @torch.no_grad()
    def encode(self, x, cond=True, normalize=None):
        # normalize = self.model_config.normalize_latent if normalize is None else normalize
        if cond:
            x_latent = self.cond_stage_model.encode(x).mode()
        else:
            x_latent = self.first_stage_model.encode(x).mode()
        # if not self.model_config.latent_before_quant_conv:
        #     x_latent = model.quant_conv(x_latent)
        # if normalize:
        #     if cond:
        #         x_latent = (x_latent - self.cond_latent_mean) / self.cond_latent_std
        #     else:
        #         x_latent = (x_latent - self.ori_latent_mean) / self.ori_latent_std
        return x_latent

    @torch.no_grad()
    def decode(self, x_latent, cond=True, normalize=None, shallow_feature = None):
        # normalize = self.model_config.normalize_latent if normalize is None else normalize
        # if normalize:
        #     if cond:
        #         x_latent = x_latent * self.cond_latent_std + self.cond_latent_mean
        #     else:
        #         x_latent = x_latent * self.ori_latent_std + self.ori_latent_mean
        # model = self.vqgan
        # if self.model_config.latent_before_quant_conv:
        #     x_latent = model.quant_conv(x_latent)
        # x_latent_quant, loss, _ = model.quantize(x_latent)
        # out = model.decode(x_latent_quant)
        if shallow_feature is not None and self.use_features:
            out = self.first_stage_model.decode(x_latent, shallow_feature)
        else:
            out = self.first_stage_model.decode(x_latent)
        return out

    @torch.no_grad()
    def sample(self, x_cond, clip_denoised=False, sample_mid_step=False):
        x_cond_latent = self.encode(x_cond, cond=True)
        if sample_mid_step:
            temp, one_step_temp = self.p_sample_loop(y=x_cond_latent,
                                                     context=self.get_cond_stage_context(x_cond),
                                                     clip_denoised=clip_denoised,
                                                     sample_mid_step=sample_mid_step)
            out_samples = []
            for i in tqdm(range(len(temp)), initial=0, desc="save output sample mid steps", dynamic_ncols=True,
                          smoothing=0.01):
                with torch.no_grad():
                    out = self.decode(temp[i].detach(), cond=False)
                out_samples.append(out.to('cpu'))

            one_step_samples = []
            for i in tqdm(range(len(one_step_temp)), initial=0, desc="save one step sample mid steps",
                          dynamic_ncols=True,
                          smoothing=0.01):
                with torch.no_grad():
                    out = self.decode(one_step_temp[i].detach(), cond=False)
                one_step_samples.append(out.to('cpu'))
            return out_samples, one_step_samples
        else:
            temp = self.p_sample_loop(y=x_cond_latent,
                                      context=self.get_cond_stage_context(x_cond),
                                      clip_denoised=clip_denoised,
                                      sample_mid_step=sample_mid_step)
            x_latent = temp
            shallow_feature = None
            if self.use_features:
                shallow_feature = self.cond_stage_model.encoder.get_features(x_cond.to(self.device))
            out = self.decode(x_latent, cond=False, shallow_feature = shallow_feature)
            return out

    @torch.no_grad()
    def sample_vqgan(self, x):
        x_rec = self.first_stage_model.decode(x)
        return x_rec
    
    @torch.no_grad()
    def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False, bs=None):
        x = super().get_input(batch, k)
        if k == self.first_stage_key:
            out = self.encode(x, cond=False)
        elif k == self.cond_stage_key:
            out = self.encode(x, cond=True)
        return out

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs):
        log = dict()
        x = batch[self.first_stage_key]
        x = x.to(memory_format=torch.contiguous_format).float()
        x_cond = batch[self.cond_stage_key]
        x_cond = x_cond.to(memory_format=torch.contiguous_format).float()
        log["inputs"] = x
        samples = self.sample(x_cond)
        log["samples"] = samples
        return log


class DiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm']

    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None):
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t)
        elif self.conditioning_key == 'concat':
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t)
        elif self.conditioning_key == 'crossattn':
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t, context=cc)
        elif self.conditioning_key == 'hybrid':
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc)
        elif self.conditioning_key == 'adm':
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)
        else:
            raise NotImplementedError()

        return out
