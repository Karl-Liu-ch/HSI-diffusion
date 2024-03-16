import sys
sys.path.append('./')
import torch.nn as nn
import torch
from torch.nn import functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.optim as optim
import torch.backends.cudnn as cudnn
from hsi_dataset import TrainDataset, ValidDataset
from utils import AverageMeter, record_loss, Loss_MRAE, Loss_RMSE, Loss_PSNR
from options import opt
import os
from torch.utils.data import DataLoader
from torch.autograd import Variable
import functools
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

class MultiAttention(nn.Module):
    def __init__(self, dimIn, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        self.dimIn = dimIn
        self.dimhidden = heads * dim_head
        self.heads = heads
        self.scale = nn.Parameter(torch.ones(heads, 1, 1))
        self.to_q = nn.Linear(dimIn, self.dimhidden, bias=False)
        self.to_k = nn.Linear(dimIn, self.dimhidden, bias=False) 
        self.to_v = nn.Linear(dimIn, self.dimhidden, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.to_out = nn.Sequential(
            nn.Linear(self.dimhidden, dimIn),
            nn.Dropout(dropout)
        )
        self.dwconv = nn.Sequential(
            nn.Conv2d(self.dimhidden, self.dimhidden, kernel_size=3, stride=1, padding=1,groups=self.dimhidden),
            nn.BatchNorm2d(self.dimhidden),
            nn.GELU()
        )
        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dimhidden, self.dimhidden // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(self.dimhidden // 4, self.dimhidden, kernel_size=1),
        )
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(self.dimhidden, self.dimhidden // 16, kernel_size=1),
            nn.BatchNorm2d(self.dimhidden // 16),
            nn.GELU(),
            nn.Conv2d(self.dimhidden // 16, 1, kernel_size=1)
        )
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dimIn, dimIn, 3, 1, 1, bias=False, groups=dimIn),
            nn.GELU(),
            nn.Conv2d(dimIn, dimIn, 3, 1, 1, bias=False, groups=dimIn),
        )
        
    def forward(self, x_in):
        b, c, h, w = x_in.shape
        x = rearrange(x_in, 'b c h w -> b (h w) c')
        batch, n, dimin = x.shape
        assert dimin == self.dimIn  
        nh = self.heads
        dk = self.dimhidden // nh
        
        q = self.to_q(x).reshape(batch, n, nh, dk).transpose(1, 2)
        k = self.to_k(x).reshape(batch, n, nh, dk).transpose(1, 2)
        v_ = self.to_v(x)
        v = v_.reshape(batch, n, nh, dk).transpose(1, 2)
        # v: [batchsize, number_heads, n, dim_heads]
        v_ = rearrange(v_, 'b (h w) d -> b d h w', h = h, w = w)
        # v_: [batchsize, dimhidden, h, w]
        
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        
        dist = torch.matmul(q, k.transpose(2,3)) * self.scale
        dist = torch.softmax(dist, dim = -1)
        dist = self.attn_drop(dist)
        att = torch.matmul(dist, v)
        att = att.transpose(1, 2).reshape(batch, n, self.dimhidden)
        
        conv_x = self.dwconv(v_)
        # Adaptive Interaction Module (AIM)
        # C-Map (before sigmoid)
        attention_reshape = att.transpose(-2,-1).contiguous().view(b, self.dimhidden, h, w)
        channel_map = self.channel_interaction(attention_reshape)
        # S-Map (before sigmoid)
        spatial_map = self.spatial_interaction(conv_x).permute(0, 2, 3, 1).contiguous().view(b, n, 1)
        # S-I
        attened_x = att * torch.sigmoid(spatial_map)
        # C-I
        conv_x = conv_x * torch.sigmoid(channel_map)
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(b, n, self.dimhidden)

        x = attened_x + conv_x

        x = self.to_out(x)

        # out = self.to_out(att)
        out = x.reshape(b, c, h, w)
        
        embed = self.pos_emb(x_in)
        
        return out + embed

class Adaptive_Channel_Attention(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            # nn.BatchNorm2d(dim // 4),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, kernel_size=1),
        )
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim, dim // 16, kernel_size=1),
            nn.BatchNorm2d(dim // 16),
            nn.GELU(),
            nn.Conv2d(dim // 16, 1, kernel_size=1)
        )

    def forward(self, x_in):
        
        B, C, H, W = x_in.shape
        N = H * W
        x = rearrange(x_in, 'b c h w -> b (h w) c')
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        v_ = v.reshape(B, C, N).contiguous().view(B, C, H, W)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # attention output
        attened_x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)

        # convolution output
        conv_x = self.dwconv(v_)

        # Adaptive Interaction Module (AIM)
        # C-Map (before sigmoid)
        attention_reshape = attened_x.transpose(-2,-1).contiguous().view(B, C, H, W)
        channel_map = self.channel_interaction(attention_reshape)
        # S-Map (before sigmoid)
        spatial_map = self.spatial_interaction(conv_x).permute(0, 2, 3, 1).contiguous().view(B, N, 1)

        # S-I
        attened_x = attened_x * torch.sigmoid(spatial_map)
        # C-I
        conv_x = conv_x * torch.sigmoid(channel_map)
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(B, N, C)

        x = attened_x + conv_x

        x = self.proj(x)
        x = self.proj_drop(x)
        
        x = rearrange(x, 'b (h w) c -> b c h w', h = H)

        return x

if __name__ == '__main__':
    multiattn = MultiAttention(31).cuda()
    input = torch.randn([1, 31, 128, 128]).cuda()
    output = multiattn(input)
    print(output.shape)