import sys
sys.path.append('./')
import torch.nn as nn
import torch
from torch.nn import functional as F
from einops import rearrange, repeat
from options import opt
import os
from utils import *
from models.transformer.MST_Plus_Plus import MS_MSA
from models.transformer.CrossViT import LePEAttentionCross
from models.transformer.CSwin import LePEAttention
from models.transformer.cat import CATBlock, partition, reverse, to_2tuple
from models.transformer.cat import Attention as CAT_Attention
from models.transformer.cat import Mlp as CAT_Mlp
from models.transformer.swin_transformer_v2 import window_partition, window_reverse
from torchsummary import summary
from models.transformer.DTN import GDFN, SGFN
from timm.scheduler.cosine_lr import CosineLRScheduler


class CSWinB(nn.Module):
    def __init__(self, dim, reso, num_heads,
                 split_size=7, qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., last_stage = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patches_resolution = reso
        self.split_size = split_size
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if self.patches_resolution == split_size:
            last_stage = True
        if last_stage:
            self.branch_num = 1
        else:
            self.branch_num = 2
        self.proj = nn.Linear(dim, dim)
        self.attns = nn.ModuleList([
            LePEAttention(
                dim//2, resolution=self.patches_resolution, idx = i,
                split_size=split_size, num_heads=num_heads//2, dim_out=dim//2,
                qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            for i in range(self.branch_num)])
        
    def forward(self, x):
        """
        x: B, H*W, C
        """
        H = W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        qkv = self.qkv(x).reshape(B, -1, 3, C).permute(2, 0, 1, 3)
        
        if self.branch_num == 2:
            x1 = self.attns[0](qkv[:,:,:,:C//2])
            x2 = self.attns[1](qkv[:,:,:,C//2:])
            attened_x = torch.cat([x1,x2], dim=2)
        else:
            attened_x = self.attns[0](qkv)
        attened_x = self.proj(attened_x)
        return attened_x

    def change_resolution(self, new_resolution: int):
        self.input_resolution = new_resolution
        self.patches_resolution = new_resolution
        for attn in self.attns:
            attn.change_resol(new_resolution)

class CSWinB_CrossAttn(nn.Module):
    def __init__(self, dim, reso, num_heads,
                 split_size=7, qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., last_stage = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patches_resolution = reso
        self.split_size = split_size
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if self.patches_resolution == split_size:
            last_stage = True
        if last_stage:
            self.branch_num = 1
        else:
            self.branch_num = 2
        self.proj = nn.Linear(2 * dim, dim)
        self.attns = nn.ModuleList([
            LePEAttentionCross(
                dim//2, resolution=self.patches_resolution, idx = i,
                split_size=split_size, num_heads=num_heads//2, dim_out=dim//2,
                qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            for i in range(self.branch_num * 2)])
        
    def forward(self, x):
        """
        x: B, H*W, C
        """
        H = W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        qkv = self.qkv(x).reshape(B, -1, 3, C).permute(2, 0, 1, 3)
        q1 = qkv[0,:,:,:C//2]
        q2 = qkv[0,:,:,C//2:]
        
        if self.branch_num == 2:
            x1 = self.attns[0](qkv[:,:,:,:C//2])
            x2 = self.attns[1](qkv[:,:,:,C//2:])
            qkv[0,:,:,C//2:] = q1
            qkv[0,:,:,:C//2] = q2
            x3 = self.attns[2](qkv[:,:,:,:C//2])
            x4 = self.attns[3](qkv[:,:,:,C//2:])
            attened_x = torch.cat([x1,x2,x3,x4], dim=2)
        attened_x = self.proj(attened_x)
        return attened_x

    def change_resolution(self, new_resolution: int):
        self.input_resolution = new_resolution
        self.patches_resolution = new_resolution
        for attn in self.attns:
            attn.change_resol(new_resolution)

class Spectral_MSAB(nn.Module):
    def __init__(self, dim, head):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.s_msa = MS_MSA(dim, dim // head, head)
        self.norm2 = nn.LayerNorm(dim)
        self.gdfn = GDFN(dim, ffn_expansion_factor=4)
    def forward(self, x, h, w):
        x_mid = self.norm1(x)
        x_mid = rearrange(x_mid, 'b (h w) c -> b h w c', h = h, w = w)
        x_mid = self.s_msa(x_mid)
        x_mid = rearrange(x_mid, 'b h w c -> b (h w) c')
        x = x + x_mid
        x = x + self.gdfn(self.norm2(x), h, w)
        return x

class Spatial_MSAB(nn.Module):
    def __init__(self, dim, head, resolution, split_size):
        super().__init__()
        self.resolution = resolution
        self.norm1 = nn.LayerNorm(dim)
        # self.cswin = CSWinB(dim, reso=resolution, num_heads=head, split_size=split_size)
        self.cswin = CSWinB_CrossAttn(dim, reso=resolution, num_heads=head, split_size=split_size)
        self.norm2 =nn.LayerNorm(dim)
        self.sgfn = SGFN(dim, dim * 4, dim)

    def forward(self, x, h, w):
        if h != self.resolution or w != self.resolution:
            self.cswin.change_resolution(h)
        x = x + self.cswin(self.norm1(x))
        x = x + self.sgfn(self.norm2(x), h, w)
        return x

class SST(nn.Module):
    def __init__(self, dim, head, resolution, split_size):
        super().__init__()
        self.resolution = resolution
        self.spectral_msa = Spectral_MSAB(dim, head)
        self.spatial_msa = Spatial_MSAB(dim, head, resolution, split_size)

    def forward(self, x):
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        if h != self.resolution or w != self.resolution:
            self.spatial_msa.cswin.change_resolution(h)
        x = self.spectral_msa(x, h, w)
        x = self.spatial_msa(x, h, w)
        x = rearrange(x, 'b (h w) c -> b c h w', h = h, w = w)
        return x

class ChannelAttn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.channel_interaction = nn.Sequential(
            nn.Conv2d(dim * 2, dim // 8, kernel_size=1, bias=False),
            # nn.BatchNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim, kernel_size=1, bias=False),
        )
    def forward(self, x):
        return self.channel_interaction(x)
    
class SSTB(nn.Module):
    def __init__(self, dim, head, resolution, split_size):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.channel_attn = ChannelAttn(dim)
        self.sst = SST(dim, head, resolution, split_size)

    def forward(self, x):
        pool = torch.cat([self.maxpool(x), self.avgpool(x)], dim=1)
        channelinter = torch.sigmoid(self.channel_attn(pool))
        channelx = x * channelinter
        x = x + self.sst(channelx)
        return x

class SSTLayer(nn.Module):
    def __init__(self, dim, head, resolution, split_size, num_blocks):
        super().__init__()
        self.model = nn.ModuleList([])
        for i in range(num_blocks):
            self.model.append(SSTB(dim, head, resolution, split_size))

    def forward(self, x):
        for layer in self.model:
            x = x + layer(x)
            # x = layer(x)
        return x

class CATLayer(nn.Module):
    """ Basic CAT layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        patch_size (int): Patch size of IPSA or CPSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value.
        qk_scale (float | None, optional): Default qk scale is head_dim ** -0.5.
        drop (float, optional): Dropout rate.
        ipsa_attn_drop (float): Attention dropout rate of InnerPatchSelfAttention.
        cpsa_attn_drop (float): Attention dropout rate of CrossPatchSelfAttention.
        drop_path (float | tuple[float], optional): Stochastic depth rate.
        norm_layer (nn.Module, optional): Normalization layer.
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer.
        use_checkpoint (bool): Whether to use checkpointing to save memory.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, patch_size, mlp_ratio=4., qkv_bias=True, 
                 qk_scale=None, drop=0., ipsa_attn_drop=0., cpsa_attn_drop=0., drop_path=0., 
                 norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.pre_ipsa_blocks = nn.ModuleList()
        self.cpsa_blocks = nn.ModuleList()
        self.post_ipsa_blocks = nn.ModuleList()
        self.spectral_blocks = nn.ModuleList()
        for i in range(depth):
            self.pre_ipsa_blocks.append(nn.ModuleList([
                CATBlock(dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, patch_size=patch_size,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                qk_scale=qk_scale, drop=drop, 
                attn_drop=ipsa_attn_drop, drop_path=drop_path,
                norm_layer=norm_layer, attn_type="ipsa", rpe=True), 
                Spectral_MSAB(dim, num_heads)
            ]))
            
            self.cpsa_blocks.append(nn.ModuleList([
                CATBlock(dim=dim, input_resolution=input_resolution,
                num_heads=1, patch_size=patch_size,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                qk_scale=qk_scale, drop=drop,
                attn_drop=cpsa_attn_drop, drop_path=drop_path,
                norm_layer=norm_layer, attn_type="cpsa", rpe=False),
                Spectral_MSAB(dim, num_heads)
            ]))
            
            self.post_ipsa_blocks.append(nn.ModuleList([
                CATBlock(dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, patch_size=patch_size,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                qk_scale=qk_scale, drop=drop, 
                attn_drop=ipsa_attn_drop, drop_path=drop_path,
                norm_layer=norm_layer, attn_type="ipsa", rpe=True), 
                Spectral_MSAB(dim, num_heads)
            ]))
            # self.spectral_blocks.append(Spectral_MSAB(dim, num_heads))
            

        # patch projection layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        b, c, h, w = x.shape
        if h != self.input_resolution[0] or w != self.input_resolution[1]:
            self.change_resolution([h, w])
        x = rearrange(x, 'b c h w -> b (h w) c')
        num_blocks = len(self.cpsa_blocks)
        for i in range(num_blocks):
            x = self.pre_ipsa_blocks[i][0](x)
            x = self.pre_ipsa_blocks[i][1](x, h, w)
            x = self.cpsa_blocks[i][0](x)
            x = self.cpsa_blocks[i][1](x, h, w)
            x = self.post_ipsa_blocks[i][0](x)
            x = self.post_ipsa_blocks[i][1](x, h, w)
        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h = h, w = w)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for i in range(self.depth):
            flops += self.pre_ipsa_blocks[i].flops()
            flops += self.cpsa_blocks[i].flops()
            flops += self.post_ipsa_blocks[i].flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

    def change_resolution(self, new_resolution):
        self.input_resolution = new_resolution
        for i in range(self.depth):
            self.pre_ipsa_blocks[i][0].change_resolution(new_resolution)
            self.cpsa_blocks[i][0].change_resolution(new_resolution)
            self.post_ipsa_blocks[i][0].change_resolution(new_resolution)

class DownSample(nn.Module):
    def __init__(self, inchannel, outchannel, pixel_shuffle = True):
        super().__init__()
        if pixel_shuffle: 
            self.model = nn.Sequential(
                nn.Conv2d(inchannel, inchannel, kernel_size=3, stride=1, padding=1), 
                nn.PixelUnshuffle(2),
                nn.Conv2d(inchannel * 4, outchannel, kernel_size=1, stride=1, padding=0, bias=False), 
            )
        else:
            self.model = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=4, stride=2, padding=1)
            )
        
    def forward(self, x):
        return self.model(x)
    
class UpSample(nn.Module):
    def __init__(self, inchannel, outchannel, pixel_shuffle = True):
        super().__init__()
        if pixel_shuffle:
            self.model = nn.Sequential(
                nn.Conv2d(inchannel, outchannel * 4, kernel_size=3, stride=1, padding=1), 
                nn.PixelShuffle(2),
            )
        else:
            self.model = nn.Sequential(
                nn.ConvTranspose2d(inchannel, outchannel, stride=2, kernel_size=2, padding=0, output_padding=0),
            )
        
    def forward(self, x):
        return self.model(x)

class SSTransformer_S(nn.Module):
    def __init__(self, in_dim = 3, out_dim = 31, hidden_dim = 32, split_size = 1, n_blocks = 4, input_resolution = [128, 128], patch_size = 8):
        super().__init__()
        self.embed = nn.Conv2d(in_dim, hidden_dim, 3, 1, 1)
        self.head = 2
        self.split_size = split_size
        self.input_resolution = input_resolution
        self.min_window = patch_size
        hidden_dim = default(hidden_dim, out_dim)
        new_ch = hidden_dim
        self.bottle_layer = nn.ModuleList([])
        for i in range(n_blocks):
            self.bottle_layer.append(CATLayer(new_ch, self.input_resolution, 1, self.head, patch_size))
        self.to_out = nn.Conv2d(new_ch, out_dim, 3, 1, 1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
    def forward(self, x):
        b, c, h_inp, w_inp = x.shape
        window = max([h_inp, w_inp])
        pad = window + (self.min_window - window % self.min_window) % self.min_window
        pad_h = pad - h_inp
        pad_w = pad - w_inp

        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')
        x = self.embed(x)
        # x_residual = x.clone()
        for layer in self.bottle_layer:
            x = x + layer(x)
        # x = self.bottle_layer(x) + x_residual
        out = self.to_out(x)

        return out[:, :, :h_inp, :w_inp]

if __name__ == '__main__':
    model = SSTransformer_S(6, 31)
    x = torch.rand([1, 6, 64, 64])
    y = model(x)
    print(y.shape)