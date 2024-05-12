import sys
sys.path.append('./')
import torch.nn as nn
import torch
from torch.nn import functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.optim as optim
import torch.backends.cudnn as cudnn
from dataset.datasets import TrainDataset, ValidDataset, TestDataset
from utils import AverageMeter, record_loss, Loss_MRAE, Loss_RMSE, Loss_PSNR, Loss_Fid, \
    Loss_SAM, Loss_SSIM, reconRGB, Loss_SID, SAM
from options import opt
import os
from utils import *
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import autograd
import functools
# from models.transformer.MST_Plus_Plus import MSAB, MS_MSA, Conv_MS_MSA
from models.gan.SpectralNormalization import SNLinear, SNConv2d, SNConvTranspose2d
from models.transformer.swin_transformer import SwinTransformerBlock
from models.transformer.swin_transformer_v2 import SwinTransformerBlock as SwinTransformerBlock_v2
from models.transformer.CSwin import CSWinBlock
from models.transformer.cat import CATBlock, partition, reverse, to_2tuple
from models.transformer.cat import Attention as CAT_Attention
from models.transformer.cat import Mlp as CAT_Mlp
from models.transformer.swin_transformer_v2 import window_partition, window_reverse
from models.transformer.agent_swin import SwinTransformerBlock as AgentSwin
from models.transformer.Base import BaseModel
from dataset.datasets import TestFullDataset
from torchsummary import summary
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
if opt.multigpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# loss function
criterion_mrae = Loss_MRAE()
criterion_rmse = Loss_RMSE()
criterion_psnr = Loss_PSNR()
criterion_psnrrgb = Loss_PSNR()
criterion_sam = Loss_SAM()
criterion_sid = Loss_SID()
criterion_fid = Loss_Fid().to(device)
criterion_ssim = Loss_SSIM().to(device)

def change_resolution(module, new_resolution):
    new_resolution = tuple(new_resolution)
    if isinstance(module, SwinTransformerBlock_v2) or isinstance(module, SwinTransformerBlock):
        module.change_resolution(new_resolution)

##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class GDFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor = 4, bias = False, act_fn = nn.GELU()):
        super(GDFN, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.norm = nn.LayerNorm(hidden_features)

        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.act_fn = act_fn

    def forward(self, x, h, w):
        x = rearrange(x, 'b (h w) c -> b c h w', h = h, w = w)
        x1, x2 = self.project_in(x).chunk(2, dim=1)
        B, C, H, W = x1.shape
        x1 = self.dwconv(self.norm(x1.view(B, C, H * W).permute(0,2,1)).permute(0,2,1).view(B,C,H,W))
        x = self.act_fn(x1) * x2
        x = self.project_out(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x

class SpatialGate(nn.Module):
    """ Spatial-Gate.
    Args:
        dim (int): Half of input channels.
    """
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim) # DW Conv

    def forward(self, x, H, W):
        # Split
        x1, x2 = x.chunk(2, dim = -1)
        B, N, C = x.shape
        x2 = self.conv(self.norm(x2).transpose(1, 2).contiguous().view(B, C//2, H, W)).flatten(2).transpose(-1, -2).contiguous()
        
        return x1 * x2

class SGFN(nn.Module):
    """ Spatial-Gate Feed-Forward Network.
    Args:
        in_features (int): Number of input channels.
        hidden_features (int | None): Number of hidden channels. Default: None
        out_features (int | None): Number of output channels. Default: None
        act_layer (nn.Module): Activation layer. Default: nn.GELU
        drop (float): Dropout rate. Default: 0.0
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.sg = SpatialGate(hidden_features//2)
        self.fc2 = nn.Linear(hidden_features//2, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        """
        Input: x: (B, H*W, C), H, W
        Output: x: (B, H*W, C)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.sg(x, H, W)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)
        return x

class CATAttnBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, patch_size=7, mlp_ratio=4., 
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.patch_size = patch_size
        if min(self.input_resolution) <= self.patch_size:
            self.patch_size = min(self.input_resolution)

        self.norm1 = norm_layer(dim)
        self.attn_local = CAT_Attention(
            dim=dim, patch_size=to_2tuple(self.patch_size), 
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, rpe=True)
        self.attn_cross = CAT_Attention(
            dim= self.patch_size ** 2, patch_size=to_2tuple(self.patch_size), 
            num_heads=1, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, rpe=False)
        self.proj = nn.Linear(dim*2, dim, bias=False)

    def forward(self, x):
        # H, W = self.input_resolution
        shortcut = x.clone()
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # partition
        patches = partition(x, self.patch_size)  # nP*B, patch_size, patch_size, C
        patches = patches.view(-1, self.patch_size * self.patch_size, C)  # nP*B, patch_size*patch_size, C

        # IPSA_pre
        attn_local = self.attn_local(patches)  # nP*B, patch_size*patch_size, C
        
        patches_cross = patches.view(B, (H // self.patch_size) * (W // self.patch_size), self.patch_size ** 2, C).permute(0, 3, 1, 2).contiguous()
        patches_cross = patches_cross.view(-1, (H // self.patch_size) * (W // self.patch_size), self.patch_size ** 2) # nP*B*C, nP*nP, patch_size*patch_size
        attn_cross = self.attn_cross(patches_cross).view(B, C, (H // self.patch_size) * (W // self.patch_size), self.patch_size ** 2)
        attn_cross = attn_cross.permute(0, 2, 3, 1).contiguous().view(-1, self.patch_size ** 2, C) # nP*B, patch_size*patch_size, C
        
        # reverse opration of partition
        attn_local = attn_local.view(-1, self.patch_size, self.patch_size, C)
        attn_cross = attn_cross.view(-1, self.patch_size, self.patch_size, C)
        x_local = reverse(attn_local, self.patch_size, H, W)  # B H' W' C
        x_local = x_local.view(B, H * W, C)
        x_cross = reverse(attn_cross, self.patch_size, H, W)  # B H' W' C
        x_cross = x_cross.view(B, H * W, C)
        x = torch.concat([x_local, x_cross], dim=-1)
        x = self.proj(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

        return x, shortcut
    
    def change_resolution(self, new_resolution):
        self.input_resolution = new_resolution

class CATLayer(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, patch_size, mlp_ratio=4., qkv_bias=True, 
                 qk_scale=None, drop=0., ipsa_attn_drop=0., cpsa_attn_drop=0., drop_path=0., 
                 norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.pre_ipsa_blocks = CATBlock(dim=dim, input_resolution=input_resolution,
                                                num_heads=num_heads, patch_size=patch_size,
                                                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                                                qk_scale=qk_scale, drop=drop, 
                                                attn_drop=ipsa_attn_drop, drop_path=drop_path,
                                                norm_layer=norm_layer, attn_type="ipsa", rpe=True)
        
        self.cpsa_blocks = CATBlock(dim=dim, input_resolution=input_resolution,
                                            num_heads=1, patch_size=patch_size,
                                            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                                            qk_scale=qk_scale, drop=drop,
                                            attn_drop=cpsa_attn_drop, drop_path=drop_path,
                                            norm_layer=norm_layer, attn_type="cpsa", rpe=False)
        
        self.post_ipsa_blocks = CATBlock(dim=dim, input_resolution=input_resolution,
                                                num_heads=num_heads, patch_size=patch_size,
                                                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                                                qk_scale=qk_scale, drop=drop, 
                                                attn_drop=ipsa_attn_drop, drop_path=drop_path,
                                                norm_layer=norm_layer, attn_type="ipsa", rpe=True)

        # patch projection layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        B, C, H, W = x.shape
        shortcut = x.clone()
        if H != self.input_resolution[0] or W != self.input_resolution[1]:
            self.change_resolution(tuple([H, W]))
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.pre_ipsa_blocks(x)
        x = self.cpsa_blocks(x)
        x = self.post_ipsa_blocks(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x, shortcut
    
    def change_resolution(self, new_resolution):
        self.input_resolution = new_resolution
        self.pre_ipsa_blocks.change_resolution(new_resolution)
        self.cpsa_blocks.change_resolution(new_resolution)
        self.post_ipsa_blocks.change_resolution(new_resolution)

class CSwin_Block(CSWinBlock):
    def forward(self, x):
        H = W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        img = self.norm1(x)
        qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3)
        
        if self.branch_num == 2:
            x1 = self.attns[0](qkv[:,:,:,:C//2])
            x2 = self.attns[1](qkv[:,:,:,C//2:])
            attened_x = torch.cat([x1,x2], dim=2)
        else:
            attened_x = self.attns[0](qkv)
        attened_x = self.proj(attened_x)
        x = x + self.drop_path(attened_x)

        return x, qkv[2]

class CSWinB(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, split_size):
        super().__init__()
        self.input_resolution = input_resolution
        self.embed = nn.Linear(dim, dim * 2, bias=False)
        self.attn = CSwin_Block(dim * 2, input_resolution[0], num_heads=num_heads * 2, split_size=split_size)
        self.out = nn.Linear(dim * 2, dim, bias=False)
        self._init_respostnorm()

    def _init_respostnorm(self):
        nn.init.constant_(self.attn.norm1.bias, 0)
        nn.init.constant_(self.attn.norm1.weight, 0)
        nn.init.constant_(self.attn.norm2.bias, 0)
        nn.init.constant_(self.attn.norm2.weight, 0)
        
    def forward(self, x):
        B, C, H, W = x.shape
        if  H != self.input_resolution[0] or W != self.input_resolution[1]:
            self.change_resolution(tuple([H, W]))
        x = rearrange(x, 'b c h w -> b (h w) c', h=H, w=W)
        x, v = self.attn(self.embed(x))
        x = self.out(x)
        v = self.out(v)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        v = rearrange(v, 'b (h w) c -> b c h w', h=H, w=W)
        return x, v
    
    def change_resolution(self, new_resolution):
        self.input_resolution = new_resolution
        self.attn.patches_resolution = new_resolution[0]
        for attn in self.attn.attns:
            attn.change_resol(new_resolution[0])

class swin_block(SwinTransformerBlock_v2):
    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        return x

class SWTB(nn.Module):
    def __init__(self, dim, input_resolution, num_heads=1, window_size=8, shift_size=0):
        super().__init__()
        self.input_resolution = input_resolution
        self.model = nn.Sequential(
            SwinTransformerBlock_v2(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size, shift_size=0),
            SwinTransformerBlock_v2(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size, shift_size=window_size // 2)
        )
        self._init_respostnorm()

    def _init_respostnorm(self):
        for blk in self.model:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)
        
    def forward(self, x):
        B, C, H, W = x.shape
        if  H != self.input_resolution[0] or W != self.input_resolution[1]:
            self.change_resolution(tuple([H, W]))
        x = rearrange(x, 'b c h w -> b (h w) c', h=H, w=W)
        x = self.model(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x
    
    def change_resolution(self, new_resolution):
        self.input_resolution = new_resolution
        self.model.apply(lambda module: change_resolution(module, new_resolution))

class Window_Attention(nn.Module):
    """ Basic attention of IPSA and CPSA.

    Args:
        dim (int): Number of input channels.
        patch_size (tuple[int]): Patch size.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value.
        qk_scale (float | None, optional): Default qk scale is head_dim ** -0.5.
        attn_drop (float, optional): Dropout ratio of attention weight.
        proj_drop (float, optional): Dropout ratio of output.
        rpe (bool): Use relative position encoding or not.
    """

    def __init__(self, dim, patch_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., rpe=True):

        super().__init__()
        self.dim = dim
        self.patch_size = patch_size  # Ph, Pw
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.rpe = rpe
        
        if self.rpe:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * patch_size[0] - 1) * (2 * patch_size[1] - 1), num_heads))  # 2*Ph-1 * 2*Pw-1, nH

            # get pair-wise relative position index for each token inside one patch
            coords_h = torch.arange(self.patch_size[0])
            coords_w = torch.arange(self.patch_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Ph, Pw
            coords_flatten = torch.flatten(coords, 1)  # 2, Ph*Pw
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Ph*Pw, Ph*Pw
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Ph*Pw, Ph*Pw, 2
            relative_coords[:, :, 0] += self.patch_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.patch_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.patch_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Ph*Pw, Ph*Pw
            self.register_buffer("relative_position_index", relative_position_index)
            trunc_normal_(self.relative_position_bias_table, std=.02)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (num_patches*B, N, C)
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] 

        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)) * self.scale
        
        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.patch_size[0] * self.patch_size[1], self.patch_size[0] * self.patch_size[1], -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, rearrange(v, 'b n h c -> b h (n c)')

def cross(x, patch_size, B, H, W, C):
    x = x.view(B, (H // patch_size) * (W // patch_size), patch_size ** 2, C).permute(0, 3, 1, 2).contiguous()
    x = x.view(-1, (H // patch_size) * (W // patch_size), patch_size ** 2)
    return x

def cross_reverse(x, patch_size, B, H, W, C):
    x = x.view(B, C, (H // patch_size) * (W // patch_size), patch_size ** 2)
    x = x.permute(0, 2, 3, 1).contiguous().view(-1, patch_size ** 2, C) # nP*B, patch_size*patch_size, C
    return x

def attn_reverse(x, patch_size, B, H, W, C):
    x = x.view(-1, patch_size, patch_size, C)
    x = reverse(x, patch_size, H, W)  # B H' W' C
    x = rearrange(x, 'b h w c -> b c h w')
    return x

class Adaptive_CATB(nn.Module):
    def __init__(self, dim=31, 
                 input_resolution=(128,128), 
                 num_heads=1, 
                 window_size=8, 
                 shift_size=0,
                 use_cswin = False, 
                 use_gdfn = False, 
                 split_size = 1, 
                 attn_type = "ipsa"):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.patch_size = window_size
        self.attn_type = attn_type
        if min(self.input_resolution) <= self.patch_size:
            self.patch_size = min(self.input_resolution)

        self.attn = Window_Attention(
            dim=dim if attn_type == "ipsa" else self.patch_size ** 2, patch_size=to_2tuple(self.patch_size), 
            num_heads=num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., rpe=True if attn_type == 'ipsa' else False)

        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.norm = nn.LayerNorm(dim)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.channel_interaction = nn.Sequential(
            nn.Conv2d(dim * 2, dim // 8, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim, kernel_size=1),
        )
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim, dim // 16, kernel_size=1),
            nn.BatchNorm2d(dim // 16),
            nn.GELU(),
            nn.Conv2d(dim // 16, 1, kernel_size=1)
        )
        self.norm2 = nn.LayerNorm(dim)
        if use_gdfn:
            self.ffn = GDFN(dim=dim, ffn_expansion_factor=4)
        else:
            self.ffn = SGFN(in_features=dim, hidden_features=dim*2)
    
    def forward(self, x_in):
        B, C, H, W = x_in.shape
        x = self.norm(rearrange(x_in, 'b c h w -> b (h w) c'))
        x = rearrange(x, 'b (h w) c -> b c h w', h = H, w = W)

        x = x.permute(0, 2, 3, 1)
        # partition
        patches = partition(x, self.patch_size)  # nP*B, patch_size, patch_size, C
        patches = patches.view(-1, self.patch_size * self.patch_size, C)  # nP*B, patch_size*patch_size, C
        # IPSA or CPSA
        if self.attn_type == "ipsa":
            attn, v = self.attn(patches)  # nP*B, patch_size*patch_size, C
        elif self.attn_type == "cpsa":
            patches = patches.view(B, (H // self.patch_size) * (W // self.patch_size), self.patch_size ** 2, C).permute(0, 3, 1, 2).contiguous()
            patches = patches.view(-1, (H // self.patch_size) * (W // self.patch_size), self.patch_size ** 2) # nP*B*C, nP*nP, patch_size*patch_size
            attn, v = self.attn(patches)
            attn = cross_reverse(attn, self.patch_size, B, H, W, C)
            v = cross_reverse(v, self.patch_size, B, H, W, C)
        else :
            raise NotImplementedError(f"Unkown Attention type: {self.attn_type}") 
        # reverse opration of partition
        attn = attn_reverse(attn, self.patch_size, B, H, W, C)
        v = attn_reverse(v, self.patch_size, B, H, W, C)
        
        convx = self.dwconv(v)
        pool = torch.concat([self.avgpool(convx), self.maxpool(convx)], dim = 1)
        channelinter = torch.sigmoid(self.channel_interaction(pool))
        spatialinter = torch.sigmoid(self.spatial_interaction(convx))
        channelx = attn * channelinter
        spatialx = convx * spatialinter
        out = self.proj(channelx + spatialx)
        out = x_in + out
        ff = self.norm2(rearrange(out, 'b c h w -> b (h w) c'))
        ff = self.ffn(ff, H, W)
        out = rearrange(ff, 'b (h w) c -> b c h w', h = H, w = W) + out
        return out
    
class Adaptive_CATLayer(nn.Module):
    def __init__(self, dim=31, 
                 input_resolution=(128,128), 
                 num_heads=1, 
                 window_size=8, 
                 shift_size=0,
                 use_cswin = False, 
                 use_gdfn = False, 
                 split_size = 1):
        super().__init__()
        self.pre_ipsa = Adaptive_CATB(dim, input_resolution, num_heads, window_size, shift_size, use_cswin, use_gdfn, split_size, attn_type='ipsa')
        self.cross_cpsa = Adaptive_CATB(dim, input_resolution, num_heads, window_size, shift_size, use_cswin, use_gdfn, split_size, attn_type='cpsa')
        self.post_ipsa = Adaptive_CATB(dim, input_resolution, num_heads, window_size, shift_size, use_cswin, use_gdfn, split_size, attn_type='ipsa')
    
    def forward(self, x):
        x = self.pre_ipsa(x)
        x = self.cross_cpsa(x)
        x = self.post_ipsa(x)
        return x

class Adaptive_SWTB(nn.Module):
    def __init__(self, dim=31, 
                 input_resolution=(128,128), 
                 num_heads=1, 
                 window_size=8, 
                 shift_size=0,
                 use_cswin = False, 
                 use_gdfn = False, 
                 split_size = 1):
        super().__init__()
        self.model = CATLayer(dim=dim, input_resolution=input_resolution, num_heads=num_heads, patch_size=window_size)

        # self.model = CATAttnBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads, patch_size=window_size)

        # self.model = CSWinB(dim=dim, 
        #                 input_resolution=input_resolution, 
        #                 num_heads=num_heads, 
        #                 split_size=split_size)
        
        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.norm = nn.LayerNorm(dim)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.channel_interaction = nn.Sequential(
            nn.Conv2d(dim * 2, dim // 8, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim, kernel_size=1),
        )
        # self.spatial_interaction = nn.Sequential(
        #     nn.Conv2d(dim, dim // 16, kernel_size=1),
        #     nn.BatchNorm2d(dim // 16),
        #     nn.GELU(),
        #     nn.Conv2d(dim // 16, 1, kernel_size=1)
        # )
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim, dim // 16, kernel_size=1),
            nn.Conv2d(dim // 16, dim // 16, 3, 1, 1, groups=dim // 16),
            nn.BatchNorm2d(dim // 16),
            nn.GELU(),
            nn.Conv2d(dim // 16, 1, 3, 1, 1, groups=1)
        )
        self.norm2 = nn.LayerNorm(dim)
        if use_gdfn:
            self.ffn = GDFN(dim=dim, ffn_expansion_factor=4)
        else:
            self.ffn = SGFN(in_features=dim, hidden_features=dim*2)
    
    def forward(self, x_in):
        B, C, H, W = x_in.shape
        x = self.norm(rearrange(x_in, 'b c h w -> b (h w) c'))
        x = rearrange(x, 'b (h w) c -> b c h w', h = H, w = W)
        B, C, H, W = x.shape
        attnx, v = self.model(x)
        convx = self.dwconv(v)
        pool = torch.concat([self.avgpool(convx), self.maxpool(convx)], dim = 1)
        channelinter = torch.sigmoid(self.channel_interaction(pool))
        spatialinter = torch.sigmoid(self.spatial_interaction(convx))
        channelx = attnx * channelinter
        spatialx = convx * spatialinter
        out = self.proj(channelx + spatialx)
        out = x_in + out
        ff = self.norm2(rearrange(out, 'b c h w -> b (h w) c'))
        ff = self.ffn(ff, H, W)
        out = rearrange(ff, 'b (h w) c -> b c h w', h = H, w = W) + out
        return out

class Conv_MS_MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Conv2d(dim_head * heads, dim, 3, 1, 1, bias=False)
        # self.pos_emb = nn.Sequential(
        #     nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        #     nn.GELU(),
        #     nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        # )
        self.dim = dim

    def forward(self, x_in):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.permute(0,3,1,2)
        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)
        v_out = v.clone()

        q_inp = rearrange(q, 'b c h w -> b (h w) c')
        k_inp = rearrange(k, 'b c h w -> b (h w) c')
        v_inp = rearrange(v, 'b c h w -> b (h w) c')
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                (q_inp, k_inp, v_inp))
        v = v
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        x = x.view(b, h, w, c).permute(0,3,1,2)
        out_c = self.proj(x).permute(0, 2, 3, 1)
        # out_p = self.pos_emb(v_inp.reshape(b,h,w,c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        # out = out_c + out_p

        return out_c, v_out

class MS_MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        # self.pos_emb = nn.Sequential(
        #     nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        #     nn.GELU(),
        #     nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        # )
        self.dim = dim

    def forward(self, x_in):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b,h*w,c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                (q_inp, k_inp, v_inp))
        v = v
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        # out_p = self.pos_emb(v_inp.reshape(b,h,w,c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        # out = out_c + out_p

        return out_c, v_inp.reshape(b,h,w,c).permute(0, 3, 1, 2)

class Adaptive_MSAB(nn.Module):
    def __init__(self, dim=31, 
                 num_blocks=2, 
                 dim_head=31, 
                 heads=1, 
                 use_conv = False, 
                 use_gdfn = False):
        super().__init__()
        self.use_conv = use_conv
        # self.model = MSAB(dim=dim, num_blocks=num_blocks, dim_head=dim_head, heads=heads, use_conv=use_conv)
        if self.use_conv:
            self.model = Conv_MS_MSA(dim=dim, dim_head=dim_head, heads=heads)
        else:
            self.model = MS_MSA(dim=dim, dim_head=dim_head, heads=heads)
        
        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.norm = nn.LayerNorm(dim)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.channel_interaction = nn.Sequential(
            nn.Conv2d(dim * 2, dim // 8, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim, kernel_size=1),
        )
        # self.spatial_interaction = nn.Sequential(
        #     nn.Conv2d(dim, dim // 16, kernel_size=1),
        #     nn.BatchNorm2d(dim // 16),
        #     nn.GELU(),
        #     nn.Conv2d(dim // 16, 1, kernel_size=1)
        # )
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim, dim // 16, kernel_size=1),
            nn.Conv2d(dim // 16, dim // 16, 3, 1, 1, groups=dim // 16),
            nn.BatchNorm2d(dim // 16),
            nn.GELU(),
            nn.Conv2d(dim // 16, 1, 3, 1, 1, groups=1)
        )
        self.norm2 = nn.LayerNorm(dim)
        if use_gdfn:
            self.ffn = GDFN(dim=dim, ffn_expansion_factor=4)
        else:
            self.ffn = SGFN(in_features=dim, hidden_features=dim*2)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.Conv2d) and self.use_conv:
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
    def forward(self, x_in):
        B, C, H, W = x_in.shape
        x = self.norm(rearrange(x_in, 'b c h w -> b (h w) c'))
        x = rearrange(x, 'b (h w) c -> b c h w', h = H, w = W)
        attnx = x.permute(0, 2, 3, 1)
        attnx, v = self.model(attnx)
        attnx = attnx.permute(0, 3, 1, 2)
        convx = self.dwconv(v)
        pool = torch.concat([self.avgpool(convx), self.maxpool(convx)], dim = 1)

        channelinter = torch.sigmoid(self.channel_interaction(pool))
        spatialinter = torch.sigmoid(self.spatial_interaction(convx))

        channelx = convx * channelinter
        spatialx = attnx * spatialinter
        out = self.proj(channelx + spatialx)
        out = x_in + out
        ff = self.norm2(rearrange(out, 'b c h w -> b (h w) c'))
        ff = self.ffn(ff, H, W)
        out = rearrange(ff, 'b (h w) c -> b c h w', h = H, w = W) + out
        return out

class DTNBlock(nn.Module):
    def __init__(self, dim, dim_head, input_resolution, num_heads, window_size, num_block, num_msab = 2, use_conv = False, use_cswin = False, use_gdfn = False, split_size = 4):
        super().__init__()
        self.dim = dim
        self.dim_head = dim_head
        self.input_resolution = tuple(input_resolution)
        self.num_heads = num_heads
        self.window_size = window_size
        layer = []
        for i in range(num_block):
            layer += [Adaptive_MSAB(dim, num_blocks=num_msab, dim_head=dim_head, heads=dim // dim_head, use_conv=use_conv, use_gdfn= use_gdfn)]
            # layer += [Adaptive_SWTB(dim, self.input_resolution, num_heads=dim // dim_head, window_size=window_size, use_cswin = use_cswin, use_gdfn= use_gdfn, split_size=split_size)]
            layer += [Adaptive_CATLayer(dim, self.input_resolution, num_heads=dim // dim_head, window_size=window_size, use_cswin = use_cswin, use_gdfn= use_gdfn, split_size=split_size)]
        self.model = nn.Sequential(*layer)
        
    def forward(self, x):
        return self.model(x)

class DownSample(nn.Module):
    def __init__(self, inchannel, outchannel):
        super().__init__()
        # self.model = nn.Sequential(
        #     nn.Conv2d(inchannel, outchannel, kernel_size=4, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(outchannel), 
        #     nn.GELU()
        # )
        # self.model = nn.Sequential(
        #     nn.Conv2d(inchannel, outchannel, kernel_size=4, stride=2, padding=1, bias=False)
        # )
        self.model = nn.Sequential(
            nn.Conv2d(inchannel, inchannel, kernel_size=3, stride=1, padding=1, bias=False), 
            nn.PixelUnshuffle(2),
            nn.Conv2d(inchannel * 4, outchannel, kernel_size=3, stride=1, padding=1, bias=False), 
        )
        
    def forward(self, x):
        return self.model(x)
    
class UpSample(nn.Module):
    def __init__(self, inchannel, outchannel):
        super().__init__()
        # self.model = nn.Sequential(
        #     nn.ConvTranspose2d(inchannel, outchannel, kernel_size=4, stride=2, padding=1),
        #     nn.BatchNorm2d(outchannel), 
        #     nn.GELU()
        # )
        # self.model = nn.Sequential(
        #     nn.ConvTranspose2d(inchannel, outchannel, stride=2, kernel_size=2, padding=0, output_padding=0),
        # )
        self.model = nn.Sequential(
            nn.Conv2d(inchannel, outchannel * 4, kernel_size=3, stride=1, padding=1, bias=False), 
            nn.PixelShuffle(2),
        )
        
    def forward(self, x):
        return self.model(x)

def ReverseTuples(tuples):
    new_tup = tuples[::-1]
    return new_tup

class DTN(nn.Module):
    def __init__(self, in_dim, 
                 out_dim,
                 img_size = [128, 128], 
                 window_size = 8, 
                 n_block=[2,2,2,2], 
                 bottleblock = 4, 
                 num_msab = 2, 
                 use_conv = False, 
                 use_cswin = False, 
                 use_gdfn = False, 
                 split_size = 1):
        super().__init__()
        self.min_window = window_size * 2 ** len(n_block)
        img_size[0] = (self.min_window - img_size[0] % self.min_window) % self.min_window + img_size[0]
        img_size[1] = (self.min_window - img_size[1] % self.min_window) % self.min_window + img_size[1]
        dim = out_dim
        self.embedding = nn.Conv2d(in_dim, dim, 3, 1, 1, bias=False)
        self.stage = len(n_block)-1
        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for num_block in n_block:
            self.encoder_layers.append(nn.ModuleList([
                DTNBlock(dim = dim_stage, 
                         dim_head = dim, 
                         input_resolution = img_size, 
                         num_heads = dim_stage // dim, 
                         window_size = window_size, 
                         num_block = num_block,
                         num_msab=num_msab, 
                         use_conv=use_conv, 
                         use_cswin = use_cswin,
                         use_gdfn = use_gdfn, 
                         split_size = split_size),
                DownSample(dim_stage, dim_stage * 2),
            ]))
            img_size[0] = img_size[0] // 2
            img_size[1] = img_size[1] // 2
            dim_stage *= 2
            split_size = min(split_size * 2, img_size[0])
        
        self.bottleneck = DTNBlock(dim = dim_stage, 
                         dim_head = dim, 
                         input_resolution = img_size, 
                         num_heads = dim_stage // dim, 
                         window_size = window_size, 
                         num_block = bottleblock,
                         num_msab=num_msab, 
                         use_conv=use_conv, 
                         use_cswin = use_cswin,
                         use_gdfn = use_gdfn, 
                         split_size = split_size)
        
        # Decoder
        self.decoder_layers = nn.ModuleList([])
        n_block.reverse()
        for num_block in n_block:
            img_size[0] = img_size[0] * 2
            img_size[1] = img_size[1] * 2
            split_size = min(split_size // 2, img_size[0])
            self.decoder_layers.append(nn.ModuleList([
                UpSample(dim_stage, dim_stage // 2), 
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),
                DTNBlock(dim = dim_stage // 2, 
                         dim_head = dim, 
                         input_resolution = img_size, 
                         num_heads = (dim_stage // 2) // dim, 
                         window_size = window_size, 
                         num_block = num_block,
                         num_msab=num_msab, 
                         use_conv=use_conv, 
                         use_cswin = use_cswin,
                         use_gdfn = use_gdfn, 
                         split_size = split_size),
            ]))
            dim_stage //= 2
        
        self.mapping = nn.Conv2d(dim, out_dim, 3, 1, 1, bias=False)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h_inp, w_inp = x.shape
        pad_h = (self.min_window - h_inp % self.min_window) % self.min_window
        pad_w = (self.min_window - w_inp % self.min_window) % self.min_window
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')
        # x = F.pad(x, [0, pad_w, 0, pad_h], mode='constant', value=0.0)

        fea = self.embedding(x)
        fea_in = fea.clone()

        # Encoder
        fea_encoder = []
        for (MSAB, FeaDownSample) in self.encoder_layers:
            fea = MSAB(fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)

        # Bottleneck
        fea = self.bottleneck(fea)

        # Decoder
        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(torch.cat([fea, fea_encoder[self.stage-i]], dim=1))
            fea = LeWinBlcok(fea)

        # Mapping
        out = self.mapping(fea) + fea_in
        # out = self.mapping(fea) + fea

        return out[:, :, :h_inp, :w_inp]
    
    def get_last_layer(self):
        return self.mapping.weight

class DTN_multi_stage(nn.Module):
    def __init__(self, *, in_dim=3, out_dim=31, n_feat=31, stage=3, img_size=[128, 128], 
                n_block=[1,1], bottleblock = 1, window = 32, num_msab = 1, **kargs):
        super(DTN_multi_stage, self).__init__()
        self.stage = stage
        self.conv_in = nn.Conv2d(in_dim, n_feat, kernel_size=3, padding=(3 - 1) // 2,bias=False)
        self.hb, self.wb = window, window
        pad_h = (self.hb - img_size[0] % self.hb) % self.hb
        pad_w = (self.wb - img_size[1] % self.wb) % self.wb
        modules_body = [DTN(in_dim=n_feat, 
                            out_dim=out_dim,
                            img_size = [img_size[0]+pad_h, img_size[1]+pad_w], 
                            window_size = 8, 
                            n_block=n_block, 
                            bottleblock = bottleblock,
                            num_msab = num_msab) for _ in range(stage)]
        self.body = nn.Sequential(*modules_body)
        self.conv_out = nn.Conv2d(n_feat * 2, out_dim, kernel_size=3, padding=(3 - 1) // 2,bias=False)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        b, c, h_inp, w_inp = x.shape
        pad_h = (self.hb - h_inp % self.hb) % self.hb
        pad_w = (self.wb - w_inp % self.wb) % self.wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')
        # x = F.pad(x, [0, pad_w, 0, pad_h], mode='constant', value=0.0)
        x = self.conv_in(x)
        h = self.body(x)
        h = self.conv_out(torch.concat([h, x], dim=1))
        # h += x
        return h[:, :, :h_inp, :w_inp]
    
    def get_last_layer(self):
        return self.conv_out.weight

def init_weights_uniform(module):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(module.weight, 1.)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
        print(f'init {module}')

if __name__ == '__main__':
    model = DTN(in_dim=3, 
                    out_dim=31,
                    img_size=[128, 128], 
                    window_size=8, 
                    n_block=[2,4], 
                    bottleblock = 4, 
                    num_msab=1, 
                    use_conv=False, 
                    use_cswin=True, 
                    use_gdfn=True).to(device)
    # model.apply(init_weights_uniform)
    # model = DTN_multi_stage(in_channels=3, out_channels=31, n_feat=31, img_size=[128, 128], window=32).to(device)
    input = torch.rand([1, 3, 128, 128]).to(device)
    output = model(input.float())
    print(output.shape)
    summary(model, (3, 256, 256))
    # spec = TrainDTN(opt, model, model_name='DTN')
    # if opt.loadmodel:
    #     try:
    #         spec.load_checkpoint()
    #     except:
    #         print('pretrained model loading failed')
    # if opt.mode == 'train':
    #     spec.train()
    #     spec.load_checkpoint(best=True)
    #     spec.test()
    #     spec.test_full_resol()
    # elif opt.mode == 'test':
    #     spec.load_checkpoint(best=True)
    #     spec.test()
    # elif opt.mode == 'testfull':
    #     spec.load_checkpoint(best=True)
    #     spec.test_full_resol()