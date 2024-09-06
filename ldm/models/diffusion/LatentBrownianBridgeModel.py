import itertools
import pdb
import random
import torch
import torch.nn as nn
from tqdm.autonotebook import tqdm

from ldm.models.diffusion.BrownianBridgeModel import BrownianBridgeModel
from ldm.modules.encoders.modules import SpatialRescaler
from models.vae.vae import PerceptualVAE

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


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class LatentBrownianBridgeModel(BrownianBridgeModel):
    def __init__(self, * , 
                 first_stage_config,
                 cond_stage_config,
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
                 epochs = 100,
                 use_features = False,
                 normalize = False,
                 ori_latent_mean = 0.0,
                 ori_latent_std = 0.0,
                 **kwargs):
        super().__init__(unet_config=unet_config, learning_rate=learning_rate, timesteps=timesteps, mt_type=mt_type, max_var=max_var, eta=eta, skip_sample=skip_sample, sample_type=sample_type, sample_step=sample_step, loss_type=loss_type, objective=objective, image_size=image_size, channels=channels, first_stage_key=first_stage_key, cond_stage_key=cond_stage_key, num_warmup=num_warmup, epochs=epochs, **kwargs)

        self.normalize = normalize
        self.cond_stage_trainable = False
        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        self.use_features = use_features
        self.ori_latent_mean = nn.Parameter(torch.tensor(ori_latent_mean), requires_grad=False)
        self.ori_latent_std = nn.Parameter(torch.tensor(ori_latent_std), requires_grad=False)

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

    def get_ema_net(self):
        return self

    def get_parameters(self):
        if self.cond_stage_key == 'SpatialRescaler':
            print("get parameters to optimize: SpatialRescaler, UNet")
            params = itertools.chain(self.denoise_fn.parameters(), self.cond_stage_model.parameters())
        else:
            print("get parameters to optimize: UNet")
            params = self.denoise_fn.parameters()
        return params

    def apply(self, weights_init):
        super().apply(weights_init)
        if self.cond_stage_model is not None:
            self.cond_stage_model.apply(weights_init)
        return self

    def get_cond_stage_context(self, x_cond):
        if self.cond_stage_model is not None:
            context = self.cond_stage_model.encode(x_cond).mode()
            if self.cond_stage_key == 'first_stage':
                context = context.detach()
        else:
            context = None
        return context

    @torch.no_grad()
    def encode(self, x, cond=True, normalize=None):
        if cond:
            x_latent = self.cond_stage_model.encode(x).mode()
            if self.normalize:
                self.cond_latent_mean = x_latent.mean()
                self.cond_latent_std = x_latent.std()
                x_latent = (x_latent - self.cond_latent_mean) / self.cond_latent_std
        else:
            x_latent = self.first_stage_model.encode(x).mode()
            if self.normalize:
                self.ori_latent_mean = nn.Parameter(self.ori_latent_mean * (self.global_step) / (self.global_step +1.0) + x_latent.mean() / (self.global_step +1.0), requires_grad=False)
                self.ori_latent_std = nn.Parameter(self.ori_latent_std * (self.global_step) / (self.global_step +1.0) + x_latent.std() / (self.global_step +1.0), requires_grad=False)
                x_latent = (x_latent - self.ori_latent_mean) / self.ori_latent_std
        return x_latent

    @torch.no_grad()
    def decode(self, x_latent, cond = False, shallow_feature = None):
        if self.use_features and shallow_feature is not None:
            if self.normalize:
                x_latent = x_latent * self.ori_latent_std + self.ori_latent_mean
            out = self.first_stage_model.decode(x_latent, shallow_feature)
        else:
            if self.normalize:
                x_latent = x_latent * self.ori_latent_std + self.ori_latent_mean
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
                shallow_feature = self.cond_stage_model.encoder.get_features(x_cond)
            out = self.decode(x_latent, cond=False, shallow_feature = shallow_feature)
            return out
        
    @torch.no_grad()
    def get_input(self, batch, k):
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
