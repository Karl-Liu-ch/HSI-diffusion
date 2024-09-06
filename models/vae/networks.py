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
# from models.vae.Base import BaseModel
from models.transformer.MST_Plus_Plus import *
from models.transformer.DT_attn import DTNBlock, UpSample, DownSample
from models.transformer.SST import SSTLayer, Spectral_MSAB
from models.transformer.SST_CAT import SSTLayer as SST_CATLayer
from models.transformer.SST_CSwin import SSTLayer as SST_CSwinLayer
from torchsummary import summary
from ldm.modules.attention import LinearAttention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

def GroupNormalize(in_channels, num_groups=31):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

class LinAttnBlock(LinearAttention):
    """to match AttnBlock usage"""
    def __init__(self, in_channels):
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels)


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = GroupNormalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

def make_attn(in_channels, attn_type="vanilla"):
    assert attn_type in ["vanilla", "linear", "none"], f'attn_type {attn_type} unknown'
    print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        return LinAttnBlock(in_channels)


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x

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
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, use_linear_attn=False, attn_type="vanilla",
                 **ignore_kwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

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
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

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
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

class CondEncoder(Encoder):
    def forward(self, x):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        cond_features = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            cond_features.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        cond_features.append(h)
        h = self.mid.block_2(h, temb)
        cond_features.append(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h, cond_features
        

class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 attn_type="vanilla", **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions. +++++++++".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
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
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
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
        print("Working with z of shape {} = {} dimensions. +++++++++".format(
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
    def __init__(self, *, ch=31, out_ch, ch_mult=(1,2,4,8), num_attn_blocks = [1,1,1,1], bottle_neck = 1,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, use_linear_attn=False, attn_type="vanilla", ckpt_path = None, 
                 **ignore_kwargs):
        super().__init__()
        assert len(ch_mult) == len(num_attn_blocks)
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.resolution = resolution
        self.in_channels = in_channels
        self.num_attn_blocks = num_attn_blocks
        self.ckpt_path = ckpt_path

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
                         num_block = num_attn_blocks[i_level],
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
                         num_block = bottle_neck,
                         num_msab=1)
        # MSAB(dim=block_in, num_blocks=num_attn_blocks, dim_head=ch, heads=block_in // ch)

        # end
        self.norm_out = GroupNormalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        
        self.apply(self._init_weights)
        if self.ckpt_path is not None:
            self.init_from_ckpt()

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, (nn.Linear, nn.Conv2d)) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
    def init_from_ckpt(self):
        sd = torch.load(self.ckpt_path, map_location='cpu')['state_dict']
        encoder_weight = {k: v for k, v in sd.items() if k.startswith("encoder.")}
        keys = []
        newkeys = []
        lenth_head = 8
        for key in encoder_weight.keys():
            keys.append(key)
            newkey = key[lenth_head:]
            newkeys.append(newkey)
        for key, newkey in zip(keys, newkeys):
            encoder_weight[newkey] = encoder_weight.pop(key)
        self.load_state_dict(encoder_weight, strict=False)
        print(f'Restore encoder from {self.ckpt_path}')

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
        # h = self.norm_out(h)
        # h = nonlinearity(h)
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
    def __init__(self, *, ch=31, out_ch, ch_mult=(1,2,4,8), num_attn_blocks = [1,1,1,1], bottle_neck = 1,
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
        print("Working with z of shape {} = {} dimensions. +++++++++".format(
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
                         num_block = bottle_neck,
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
                         num_block = num_attn_blocks[i_level],
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

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, (nn.Linear, nn.Conv2d)) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
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

        # h = self.norm_out(h)
        # h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class DualTransformerDecoderCond(nn.Module):
    def __init__(self, *, ch=31, out_ch, ch_mult=(1,2,4,8), num_attn_blocks = [1,1,1,1], bottle_neck = 1,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 attn_type="vanilla", use_cond = False, **ignorekwargs):
        super().__init__()
        assert len(ch_mult) == len(num_attn_blocks)
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_attn_blocks = num_attn_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        self.use_cond = use_cond

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions. +++++++++".format(
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
                         num_block = bottle_neck,
                         num_msab=1)
        # MSAB(dim=block_in, num_blocks=num_attn_blocks, dim_head=ch, heads=block_in // ch)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            block += [nn.Conv2d(block_in, block_out, 3, 1, 1)]
            block_in = block_out
            if self.use_cond:
                block += [nn.Conv2d(block_in * 2, block_in, 1, 1, 0, bias=False)]
            attn = DTNBlock(dim = block_in, 
                         dim_head = ch, 
                         input_resolution = [curr_res, curr_res], 
                         num_heads = block_out // ch, 
                         window_size = 8, 
                         num_block = num_attn_blocks[i_level],
                         num_msab=1)
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = UpSample(block_in, block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = GroupNormalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, (nn.Linear, nn.Conv2d)) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
    def forward(self, z, cond = None):
        #assert z.shape[1:] == self.z_shape[1:]
        if self.use_cond:
            assert cond is not None, 'cond is None'
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.attn_1(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            if self.use_cond:
                h = self.up[i_level].block[0](h)
                h = self.up[i_level].block[1](torch.concat([h, cond[i_level]], dim=1))
                h = self.up[i_level].attn(h)
            else:
                h = self.up[i_level].block[0](h)
                h = self.up[i_level].attn(h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        # h = self.norm_out(h)
        # h = nonlinearity(h)
        h = self.conv_out(h)
        return h

class Spectral_MSAB_Layer(nn.Module):
    def __init__(self, dim, heads, n_blocks) -> None:
        super().__init__()
        self.layer = nn.ModuleList([])
        for i in range(n_blocks):
            self.layer.append(Spectral_MSAB(dim, heads))

    def forward(self, x):
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        for layer in self.layer:
            x = layer(x, h, w)
        x = rearrange(x, 'b (h w) c -> b c h w', h = h, w = w)
        return x


class SSTEncoder(nn.Module):
    def __init__(self, *, ch=31, out_ch, ch_mult=(1,2,4,8), num_attn_blocks = [1,1,1,1], bottle_neck = 1,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, use_linear_attn=False, attn_type="vanilla", ckpt_path = None, 
                 **ignore_kwargs):
        super().__init__()
        assert len(ch_mult) == len(num_attn_blocks)
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.resolution = resolution
        self.in_channels = in_channels
        self.num_attn_blocks = num_attn_blocks
        self.ckpt_path = ckpt_path

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
        split_size = 1
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
            attn = Spectral_MSAB_Layer(dim=block_in, heads=block_in // ch, n_blocks=num_attn_blocks[i_level])
            # attn = SSTLayer(dim = block_in, 
            #              head = block_in // ch * 2, 
            #              resolution = [curr_res, curr_res], 
            #              split_size=split_size, 
            #              num_blocks = num_attn_blocks[i_level])
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = DownSample(block_in, block_in)
                curr_res = curr_res // 2
                split_size *= 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()

        self.mid.attn_1 = Spectral_MSAB_Layer(dim=block_in, heads=block_in // ch, n_blocks=bottle_neck)
        # self.mid.attn_1 = SSTLayer(dim = block_in, 
        #                  head = block_in // ch * 2, 
        #                  resolution = [curr_res, curr_res], 
        #                  split_size=split_size, 
        #                  num_blocks = bottle_neck)

        # end
        self.norm_out = GroupNormalize(block_in, num_groups=block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        
        self.apply(self._init_weights)
        if self.ckpt_path is not None:
            self.init_from_ckpt()

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, (nn.Linear, nn.Conv2d)) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        

    def init_from_ckpt(self):
        sd = torch.load(self.ckpt_path, map_location='cpu')['state_dict']
        encoder_weight = {k: v for k, v in sd.items() if k.startswith("encoder.")}
        keys = []
        newkeys = []
        lenth_head = 8
        for key in encoder_weight.keys():
            keys.append(key)
            newkey = key[lenth_head:]
            newkeys.append(newkey)
        for key, newkey in zip(keys, newkeys):
            encoder_weight[newkey] = encoder_weight.pop(key)
        self.load_state_dict(encoder_weight, strict=False)

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
        # h = self.norm_out(h)
        # h = nonlinearity(h)
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

class SSTDecoder(nn.Module):
    def __init__(self, *, ch=31, out_ch, ch_mult=(1,2,4,8), num_attn_blocks = [1,1,1,1], bottle_neck = 1,
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
        split_size = 2 ** (self.num_resolutions - 1)

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions. +++++++++".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.attn_1 = Spectral_MSAB_Layer(dim=block_in, heads=block_in // ch, n_blocks=bottle_neck)
        # self.mid.attn_1 = SSTLayer(dim = block_in, 
        #                  head = block_in // ch * 2, 
        #                  resolution = [curr_res, curr_res], 
        #                  split_size=split_size, 
        #                  num_blocks = num_attn_blocks[-1])

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
            attn = Spectral_MSAB_Layer(dim=block_in, heads=block_in // ch, n_blocks=num_attn_blocks[i_level])
            # attn = SSTLayer(dim = block_in, 
            #              head = block_in // ch * 2, 
            #              resolution = [curr_res, curr_res], 
            #              split_size=split_size, 
            #              num_blocks = num_attn_blocks[i_level])
            # MSAB(dim=block_in, num_blocks=num_attn_blocks, dim_head=ch, heads=block_in // ch)
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = UpSample(block_in, block_in)
                # nn.ConvTranspose2d(block_in, block_in, kernel_size=4, stride=2, padding=1)
                curr_res = curr_res * 2
                split_size = split_size // 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = GroupNormalize(block_in, block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, (nn.Linear, nn.Conv2d)) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        

    def forward(self, z):
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
        h = self.conv_out(h)
        return h


if __name__ == '__main__':
    encoder = DualTransformerEncoder(ch=31, out_ch=6, resolution=128, ch_mult=(1,2,4,4), num_attn_blocks=[1,1,1,1], bottle_neck=1, attn_resolutions=8, in_channels=6, z_channels=64)
    input = torch.rand([1, 6, 128, 128])
    output = encoder(input)
    print(output.shape)
    decoder = DualTransformerDecoderCond(ch=31, out_ch=31, resolution=128, ch_mult=(1,2,4,4), num_attn_blocks=[1,1,1,1], bottle_neck=1, attn_resolutions=8, in_channels=31, z_channels=64, use_cond=True)
    noise = torch.rand([1, 64, 16, 16])
    cond = encoder.get_features(input)
    output = decoder(noise, cond)
    print(output.shape)