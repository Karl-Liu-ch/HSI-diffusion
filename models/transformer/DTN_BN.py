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
from models.transformer.MST_Plus_Plus import MSAB
from models.transformer.SN_MST_Plus_Plus import MSAB as SN_MSAB
from models.gan.SpectralNormalization import SNLinear, SNConv2d, SNConvTranspose2d
from models.transformer.swin_transformer import SwinTransformerBlock
from models.transformer.swin_transformer_v2 import SwinTransformerBlock as SwinTransformerBlock_v2
from models.transformer.CSwin import CSWinBlock
from models.transformer.sn_swin_transformer_v2 import SwinTransformerBlock as SN_SwinTransformerBlock
from models.transformer.agent_swin import SwinTransformerBlock as AgentSwin
from models.transformer.Base import BaseModel
from dataset.datasets import TestFullDataset
from torchsummary import summary
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

class CSWinB(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, split_size):
        super().__init__()
        self.input_resolution = input_resolution
        self.embed = nn.Linear(dim, dim * 2, bias=False)
        self.attn = CSWinBlock(dim * 2, input_resolution[0], num_heads=num_heads * 2, split_size=split_size)
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
        x = self.out(self.attn(self.embed(x)))
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x
    
    def change_resolution(self, new_resolution):
        self.input_resolution = new_resolution
        self.attn.patches_resolution = new_resolution[0]
        for attn in self.attn.attns:
            attn.change_resol(new_resolution[0])

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
        if use_cswin:
            self.model = CSWinB(dim=dim, 
                            input_resolution=input_resolution, 
                            num_heads=num_heads, 
                            split_size=split_size)
        else:
            self.model = SWTB(dim=dim, 
                            input_resolution=input_resolution, 
                            num_heads=num_heads, 
                            window_size=window_size,
                            shift_size=shift_size)
        
        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.norm = nn.LayerNorm(dim)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.proj = nn.Linear(dim, dim)
        self.channel_interaction = nn.Sequential(
            nn.Conv2d(dim * 2, dim // 8, kernel_size=1),
            nn.BatchNorm2d(dim // 8),
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
        B, C, H, W = x.shape
        attnx = self.model(x)
        convx = self.dwconv(x)
        pool = torch.concat([self.avgpool(convx), self.maxpool(convx)], dim = 1)

        channelinter = torch.sigmoid(self.channel_interaction(pool))
        spatialinter = torch.sigmoid(self.spatial_interaction(attnx))
        
        channelx = attnx * channelinter
        spatialx = convx * spatialinter
        out = self.proj(rearrange(channelx, 'b c h w -> b (h w) c') + rearrange(spatialx, 'b c h w -> b (h w) c'))
        out = rearrange(out, 'b (h w) c -> b c h w', h = H, w = W)
        out = x_in + out
        ff = self.norm2(rearrange(out, 'b c h w -> b (h w) c'))
        ff = self.ffn(ff, H, W)
        out = rearrange(ff, 'b (h w) c -> b c h w', h = H, w = W) + out
        return out

class Adaptive_MSAB(nn.Module):
    def __init__(self, dim=31, 
                 num_blocks=2, 
                 dim_head=31, 
                 heads=1, 
                 use_conv = False, 
                 use_gdfn = False):
        super().__init__()
        self.use_conv = use_conv
        self.model = MSAB(dim=dim, num_blocks=num_blocks, dim_head=dim_head, heads=heads, use_conv=use_conv)
        
        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.norm = nn.LayerNorm(dim)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.proj = nn.Linear(dim, dim)
        self.channel_interaction = nn.Sequential(
            nn.Conv2d(dim * 2, dim // 8, kernel_size=1),
            nn.BatchNorm2d(dim // 8),
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
        attnx = self.model(x)
        convx = self.dwconv(x)
        pool = torch.concat([self.avgpool(attnx), self.maxpool(attnx)], dim = 1)

        channelinter = torch.sigmoid(self.channel_interaction(pool))
        spatialinter = torch.sigmoid(self.spatial_interaction(convx))

        channelx = convx * channelinter
        spatialx = attnx * spatialinter
        out = self.proj(rearrange(channelx, 'b c h w -> b (h w) c') + rearrange(spatialx, 'b c h w -> b (h w) c'))
        out = rearrange(out, 'b (h w) c -> b c h w', h = H, w = W)
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
            layer += [Adaptive_SWTB(dim, self.input_resolution, num_heads=dim // dim_head, window_size=window_size, use_cswin = use_cswin, use_gdfn= use_gdfn, split_size=split_size)]
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
        if b == 1:
            self.apply(freeze_norm_stats)
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
                    num_msab=1).to(device)
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