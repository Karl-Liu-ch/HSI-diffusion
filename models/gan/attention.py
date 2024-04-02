import sys
sys.path.append('./')
import torch.nn as nn
import torch
from torch.nn import functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.optim as optim
import torch.backends.cudnn as cudnn
from options import opt
import os
from torch import autograd
import functools
from models.gan.SpectralNormalization import *

class ChannelAttention(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(*self.conv(input_channels, reduction_ratio))

    def conv(self, input_channels, reduction_ratio=16):
        layer = [nn.Conv2d(input_channels, input_channels // reduction_ratio, 1, bias=False),
                nn.ReLU(),
                nn.Conv2d(input_channels // reduction_ratio, input_channels, 1, bias=False),
                nn.Sigmoid()]
        return layer
    
    def forward(self, x):
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        avg_out = self.fc(avg_out)
        max_out = self.fc(max_out)
        return avg_out * x + max_out * x
    
class SpatialAttention(nn.Module):
    def __init__(self, dim, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.Conv = nn.Sequential(*self.conv(dim = dim, kernel_size = kernel_size))
        self.sigmoid = nn.Sigmoid()

    def conv(self, dim, kernel_size):
        layer = [nn.Conv2d(dim, dim // 16, kernel_size, padding=(kernel_size - 1) // 2),
                nn.ReLU(),
                nn.Conv2d(dim // 16, 1,  kernel_size, padding=(kernel_size - 1) // 2)]
        return layer
    
    def forward(self, x_in):
        x = self.Conv(x_in)
        return self.sigmoid(x) * x_in

class SN_ChannelAttention(ChannelAttention):
    def conv(self, input_channels, reduction_ratio=16):
        layer = [SNConv2d(input_channels, input_channels // reduction_ratio, 1, bias=False),
                nn.ReLU(),
                SNConv2d(input_channels // reduction_ratio, input_channels, 1, bias=False),
                nn.Sigmoid()]
        return layer

class SN_SpatialAttention(SpatialAttention):
    def conv(self, dim, kernel_size):
        layer = [SNConv2d(dim, dim // 16, kernel_size, padding=(kernel_size - 1) // 2),
                nn.ReLU(),
                SNConv2d(dim // 16, 1,  kernel_size, padding=(kernel_size - 1) // 2)]
        return layer

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

def shift_back(inputs,step=2):          # input [bs,28,256,310]  output [bs, 28, 256, 256]
    [bs, nC, row, col] = inputs.shape
    down_sample = 256//row
    step = float(step)/float(down_sample*down_sample)
    out_col = row
    for i in range(nC):
        inputs[:,i,:,:out_col] = \
            inputs[:,i,:,int(step*i):int(step*i)+out_col]
    return inputs[:, :, :, :out_col]

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
        self.to_q = SNLinear(dim, dim_head * heads, bias=False)
        self.to_k = SNLinear(dim, dim_head * heads, bias=False)
        self.to_v = SNLinear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = SNLinear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            SNConv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            nn.ReLU(),
            SNConv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
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
        out_p = self.pos_emb(v_inp.reshape(b,h,w,c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            SNConv2d(dim, dim * mult, 1, 1, bias=False),
            nn.ReLU(),
            SNConv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            nn.ReLU(),
            SNConv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)

class SN_MSAB(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
            num_blocks,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                MS_MSA(dim=dim, dim_head=dim_head, heads=heads),
                FeedForward(dim=dim)
            ]))

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out

class AttentionBlock(nn.Module):
    def __init__(self, input_nums, kernalsize=7, reduction_ratio=16):
        super().__init__()
        self.blocks = []
        self.blocks += self.Block(input_nums, kernalsize, reduction_ratio)
        self.net = nn.Sequential(*self.blocks)
    
    def Block(self, input_nums, kernalsize, reduction_ratio):
        layer = []
        layer += [ResnetBlock(input_nums, padding_type='reflect', use_dropout=False, use_bias=True), 
                  ChannelAttention(input_nums, reduction_ratio)]
        return layer
    
    def forward(self, x):
        x_ch = self.net(x)
        return x_ch

class SN_AttentionBlock(AttentionBlock):
    def Block(self, input_nums, kernalsize, reduction_ratio):
        layer = []
        layer += [SNResnetBlock(input_nums, padding_type='reflect', use_dropout=False, use_bias=True), 
                  SN_ChannelAttention(input_nums, reduction_ratio)]
        return layer       

class AttentionDiscirminator(nn.Module):
    def __init__(self, input_nums, n_layer=3):
        super().__init__()
        self.blocks = []
        inchannels = input_nums
        for i in range(n_layer):
            outchannels = inchannels * 2
            self.blocks += self.Block(inchannels, outchannels)
            inchannels = outchannels
        self.blocks += self.conv(input_channels = inchannels, output_channels = 1, kernel_size = 4, stride = 1, padding = 0)
        self.blocks += [nn.AdaptiveAvgPool2d((1, 1)), 
                        nn.Flatten()]
        self.net = nn.Sequential(*self.blocks)
    
    def conv(self, input_channels, output_channels, kernel_size, stride, padding, bias = True):
        Layer = []
        Layer.append(nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        return Layer
    
    def Block(self, input_nums, output_nums):
        layer = []
        layer += [AttentionBlock(input_nums)]
        layer += self.conv(input_nums, output_nums, (3,3), (2,2), (1,1))
        layer += [nn.ReLU(True)]
        return layer
    
    def forward(self, x):
        return self.net(x)

class SN_AttentionDiscirminator(AttentionDiscirminator):
    def conv(self, input_channels, output_channels, kernel_size, stride, padding, bias=True):
        Layer = [SNConv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
        return Layer
    
    def Block(self, input_nums, output_nums):
        layer = []
        layer += [SN_AttentionBlock(input_nums)]
        layer += self.conv(input_nums, output_nums, (3,3), (2,2), (1,1))
        layer += [nn.ReLU(True)]
        return layer

class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type = 'reflect', norm_layer = nn.BatchNorm2d, use_dropout = False, use_bias = False):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out
if __name__ == '__main__':
    model = SN_AttentionDiscirminator(input_nums=34).cuda()
    input = torch.rand([1, 34, 128, 128]).cuda()
    output = model(input)
    print(output.shape)