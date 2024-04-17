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
from models.transformer.DTN import DTNBlock, UpSample, DownSample
from torchsummary import summary

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
if opt.multigpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

def GroupNormalize(in_channels, num_groups=31):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = GroupNormalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = GroupNormalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h

class Encoder(nn.Module):
    def __init__(self, *, ch=31, out_ch, ch_mult=(1,2,4,8), num_attn_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, use_linear_attn=False, attn_type="vanilla",
                 **ignore_kwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.resolution = resolution
        self.in_channels = in_channels
        self.num_attn_blocks = num_attn_blocks

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            block = ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout)
            block_in = block_out
            attn = MSAB(dim=block_in, num_blocks=num_attn_blocks, dim_head=ch, heads=block_in // ch)
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = nn.Conv2d(block_in, block_in, kernel_size=3, stride=2, padding=1)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.attn_1 = MSAB(dim=block_in, num_blocks=num_attn_blocks, dim_head=ch, heads=block_in // ch)

        # end
        self.norm_out = GroupNormalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            h = self.down[i_level].block(hs[-1], temb)
            hs.append(h)
            h = self.down[i_level].attn(hs[-1])
            hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.attn_1(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

    def get_features(self, x):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        features = []
        for i_level in range(self.num_resolutions):
            h = self.down[i_level].block(hs[-1], temb)
            hs.append(h)
            h = self.down[i_level].attn(hs[-1])
            hs.append(h)
            features.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.attn_1(h)
        # hs.append(h)

        # end
        h = self.norm_out(h)
        # hs.append(h)
        # h = nonlinearity(h)
        # h = self.conv_out(h)
        return features

class CondEncoder(Encoder):
    def forward(self, x):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        cond_features = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            h = self.down[i_level].block(hs[-1], temb)
            hs.append(h)
            h = self.down[i_level].attn(hs[-1])
            hs.append(h)
            cond_features.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.attn_1(h)
        cond_features.append(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h, cond_features
        
class Decoder(nn.Module):
    def __init__(self, *, ch=31, out_ch, ch_mult=(1,2,4,8), num_attn_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 attn_type="vanilla", **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_attn_blocks = num_attn_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.attn_1 = MSAB(dim=block_in, num_blocks=num_attn_blocks, dim_head=ch, heads=block_in // ch)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            block = ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout)
            block_in = block_out
            attn = MSAB(dim=block_in, num_blocks=num_attn_blocks, dim_head=ch, heads=block_in // ch)
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = nn.ConvTranspose2d(block_in, block_in, kernel_size=4, stride=2, padding=1)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = GroupNormalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.attn_1(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            h = self.up[i_level].block(h, temb)
            h = self.up[i_level].attn(h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h

class FirstStageDecoder(nn.Module):
    def __init__(self, *, ch=31, out_ch, ch_mult=(1,2,4,8), num_attn_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_attn_blocks = num_attn_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels * 2,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.conv_1 = nn.Conv2d(in_channels=block_in*2, out_channels=block_in*2, kernel_size=1, stride=1)
        self.mid.attn_1 = MSAB(dim=block_in*2, num_blocks=num_attn_blocks, dim_head=ch, heads=block_in*2 // ch) 
        self.mid.block_1 = ResnetBlock(in_channels=block_in*2,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            conv = nn.ModuleList()
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            conv = nn.Conv2d(in_channels=block_in+block_out, out_channels=block_in+block_out, kernel_size=1, stride=1)
            block = ResnetBlock(in_channels=block_in+block_out,
                                    out_channels=block_out,
                                    temb_channels=self.temb_ch,
                                    dropout=dropout)
            block_in = block_out
            attn = MSAB(dim=block_in, num_blocks=num_attn_blocks, dim_head=ch, heads=block_in // ch)
            up = nn.Module()
            up.conv = conv
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = nn.ConvTranspose2d(block_in, block_in, kernel_size=4, stride=2, padding=1)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.conv_final = nn.Conv2d(in_channels=block_in*2, out_channels=block_in, kernel_size=1, stride=1)
        self.norm_out = GroupNormalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z, cond_features):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        # z = torch.concat([z, cond], dim=1)
        h = self.conv_in(z)

        # middle
        h = self.mid.conv_1(torch.concat([h, cond_features[-1]], dim=1))
        h = self.mid.attn_1(h)
        h = self.mid.block_1(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            cond_feature = cond_features[i_level+1]
            # print(cond_feature.shape, h.shape)
            h = self.up[i_level].conv(torch.concat([h, cond_feature], dim=1))
            h = self.up[i_level].block(h, temb)
            h = self.up[i_level].attn(h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h
        
        h = self.conv_final(torch.concat([h, cond_features[0]], dim=1))
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean


class DualTransformerEncoder(nn.Module):
    def __init__(self, *, ch=31, out_ch, ch_mult=(1,2,4,8), num_attn_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, use_linear_attn=False, attn_type="vanilla",
                 **ignore_kwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.resolution = resolution
        self.in_channels = in_channels
        self.num_attn_blocks = num_attn_blocks

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            block = nn.Conv2d(block_in, block_out, 3, 1, 1)
            # ResnetBlock(in_channels=block_in,
            #                     out_channels=block_out,
            #                     temb_channels=self.temb_ch,
            #                     dropout=dropout)
            block_in = block_out
            attn = DTNBlock(dim = block_in, 
                         dim_head = ch, 
                         input_resolution = [curr_res, curr_res], 
                         num_heads = block_in // ch, 
                         window_size = 8, 
                         num_block = 1,
                         num_msab=1)
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = DownSample(block_in, block_in)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.attn_1 = DTNBlock(dim = block_in, 
                         dim_head = ch, 
                         input_resolution = [curr_res, curr_res], 
                         num_heads = block_in // ch, 
                         window_size = 8, 
                         num_block = 1,
                         num_msab=1)
        # MSAB(dim=block_in, num_blocks=num_attn_blocks, dim_head=ch, heads=block_in // ch)

        # end
        self.norm_out = GroupNormalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            h = self.down[i_level].block(hs[-1])
            hs.append(h)
            h = self.down[i_level].attn(hs[-1])
            hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.attn_1(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

    def get_features(self, x):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        features = []
        for i_level in range(self.num_resolutions):
            h = self.down[i_level].block(hs[-1])
            hs.append(h)
            h = self.down[i_level].attn(hs[-1])
            hs.append(h)
            features.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.attn_1(h)
        # hs.append(h)

        # end
        h = self.norm_out(h)
        # hs.append(h)
        # h = nonlinearity(h)
        # h = self.conv_out(h)
        return features

class DualTransformerDecoder(nn.Module):
    def __init__(self, *, ch=31, out_ch, ch_mult=(1,2,4,8), num_attn_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 attn_type="vanilla", **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_attn_blocks = num_attn_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.attn_1 = DTNBlock(dim = block_in, 
                         dim_head = ch, 
                         input_resolution = [curr_res, curr_res], 
                         num_heads = block_in // ch, 
                         window_size = 8, 
                         num_block = 1,
                         num_msab=1)
        # MSAB(dim=block_in, num_blocks=num_attn_blocks, dim_head=ch, heads=block_in // ch)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            block = nn.Conv2d(block_in, block_out, 3, 1, 1)
            # ResnetBlock(in_channels=block_in,
            #                              out_channels=block_out,
            #                              temb_channels=self.temb_ch,
            #                              dropout=dropout)
            block_in = block_out
            attn = DTNBlock(dim = block_in, 
                         dim_head = ch, 
                         input_resolution = [curr_res, curr_res], 
                         num_heads = block_out // ch, 
                         window_size = 8, 
                         num_block = 1,
                         num_msab=1)
            # MSAB(dim=block_in, num_blocks=num_attn_blocks, dim_head=ch, heads=block_in // ch)
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = UpSample(block_in, block_in)
                # nn.ConvTranspose2d(block_in, block_in, kernel_size=4, stride=2, padding=1)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = GroupNormalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.attn_1(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            h = self.up[i_level].block(h)
            h = self.up[i_level].attn(h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h

if __name__ == '__main__':
    model = DualTransformerEncoder(ch=31, out_ch=31, resolution=128, num_attn_blocks=1, attn_resolutions=8, in_channels=31, z_channels=64).cuda()
    input = torch.rand([1, 31, 128, 128]).cuda()
    output = model(input)
    print(output.shape)
    summary(model, (31, 128, 128))