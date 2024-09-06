import sys
sys.path.append('./')
from models.transformer.attention import *
from models.transformer.cat import partition, reverse
import numpy as np
from models.transformer.CSwin import img2windows, windows2img, DropPath, Mlp


class LePEAttentionCross(nn.Module):
    def __init__(self, dim, resolution, idx, split_size=7, dim_out=None, num_heads=8, attn_drop=0., proj_drop=0., qk_scale=None):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        # self.scale = qk_scale or head_dim ** -0.5
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.idx = idx
        if idx == -1:
            H_sp, W_sp = self.resolution, self.resolution
        elif idx == 0:
            H_sp, W_sp = self.resolution, self.split_size
        elif idx == 1:
            W_sp, H_sp = self.resolution, self.split_size
        elif idx == 2:
            H_sp, W_sp = self.resolution, self.split_size
        elif idx == 3:
            W_sp, H_sp = self.resolution, self.split_size
        else:
            print ("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp
        stride = 1
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim)

        self.attn_drop = nn.Dropout(attn_drop)

    def change_resol(self, new_resolution):
        self.resolution = new_resolution
        if self.idx == -1:
            H_sp, W_sp = new_resolution, new_resolution
        elif self.idx == 0:
            H_sp, W_sp = new_resolution, self.split_size
        elif self.idx == 1:
            W_sp, H_sp = new_resolution, self.split_size
        elif self.idx == 2:
            H_sp, W_sp = self.resolution, self.split_size
        elif self.idx == 3:
            W_sp, H_sp = self.resolution, self.split_size
        else:
            print ("ERROR MODE", self.idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp

    def im2cswin(self, x):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp* self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def im2cswin_cross(self, x):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.W_sp, self.H_sp)
        x = x.reshape(-1, self.H_sp* self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def get_lepe(self, x, func):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)

        H_sp, W_sp = self.H_sp, self.W_sp
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp) ### B', C, H', W'

        lepe = func(x) ### B', C, H', W'
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()

        x = x.reshape(-1, self.num_heads, C // self.num_heads, self.H_sp* self.W_sp).permute(0, 1, 3, 2).contiguous()
        return x, lepe

    def forward(self, qkv, cross = False):
        """
        x: B L C
        """
        q,k,v = qkv[0], qkv[1], qkv[2]

        ### Img2Window
        H = W = self.resolution
        B, L, C = q.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        if cross:
            q = self.im2cswin_cross(q)
        else:
            q = self.im2cswin(q)
        k = self.im2cswin(k)
        v, lepe = self.get_lepe(v, self.get_v)

        # q = q * self.scale
        # attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)) * self.scale  # B head N C @ B head C N --> B head N N
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v) + lepe
        x = x.transpose(1, 2).reshape(-1, self.H_sp* self.W_sp, C)  # B head N N @ B head N C

        ### Window2Img
        x = windows2img(x, self.H_sp, self.W_sp, H, W).view(B, -1, C)  # B H' W' C

        return x

class CSWinCrossAttention(nn.Module):
    def __init__(self, dim, resolution, idx, split_size=7, dim_out=None, num_heads=8, attn_drop=0., proj_drop=0., qk_scale=None, qkv_bias = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.resolution = resolution
        self.split_size = split_size
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.idx = idx
        if idx == -1:
            H_sp, W_sp = self.resolution, self.resolution
        elif idx == 0:
            H_sp, W_sp = self.resolution, self.split_size
        elif idx == 1:
            W_sp, H_sp = self.resolution, self.split_size
        elif idx == 2:
            H_sp, W_sp = self.resolution, self.split_size
        elif idx == 3:
            W_sp, H_sp = self.resolution, self.split_size
        else:
            print ("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp
        self.to_q = nn.Linear(dim, dim, bias = qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias = qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias = qkv_bias)
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def im2cswin(self, x):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp* self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def im2cswin_cross(self, x):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.W_sp, self.H_sp)
        x = x.reshape(-1, self.H_sp* self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def get_lepe(self, x, func):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)

        H_sp, W_sp = self.H_sp, self.W_sp
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp) ### B', C, H', W'

        lepe = func(x) ### B', C, H', W'
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()

        x = x.reshape(-1, self.num_heads, C // self.num_heads, self.H_sp* self.W_sp).permute(0, 1, 3, 2).contiguous()
        return x, lepe

    def forward(self, x1, x2):
        B, N, C = x1.shape
        q = self.to_q(x1)
        v = self.to_k(x2)
        k = self.to_v(x2)
        q = self.im2cswin_cross(q)
        k = self.im2cswin(k)
        v, lepe = self.get_lepe(v, self.get_v)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = nn.functional.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v + lepe).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    def change_resol(self, new_resolution):
        self.resolution = new_resolution
        if self.idx == -1:
            H_sp, W_sp = new_resolution, new_resolution
        elif self.idx == 0:
            H_sp, W_sp = new_resolution, self.split_size
        elif self.idx == 1:
            W_sp, H_sp = new_resolution, self.split_size
        elif self.idx == 2:
            H_sp, W_sp = self.resolution, self.split_size
        elif self.idx == 3:
            W_sp, H_sp = self.resolution, self.split_size
        else:
            print ("ERROR MODE", self.idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp


class CSWinBlock(nn.Module):

    def __init__(self, dim, reso, num_heads,
                 split_size=7, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 last_stage=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patches_resolution = reso
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm1 = norm_layer(dim)

        if self.patches_resolution == split_size:
            last_stage = True
        if last_stage:
            self.branch_num = 1
        else:
            self.branch_num = 2
        self.proj = nn.Linear(dim * 2, dim)
        self.proj_drop = nn.Dropout(drop)
        
        if last_stage:
            self.attns = nn.ModuleList([
                LePEAttentionCross(
                    dim, resolution=self.patches_resolution, idx = -1,
                    split_size=split_size, num_heads=num_heads, dim_out=dim,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])
        else:
            self.attns = nn.ModuleList([
                LePEAttentionCross(
                    dim//2, resolution=self.patches_resolution, idx = i,
                    split_size=split_size, num_heads=num_heads//2, dim_out=dim//2,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])
            self.crossattns = nn.ModuleList([CSWinCrossAttention(dim // 2, resolution=self.patches_resolution, idx=i, split_size=split_size, 
                                                            num_heads=num_heads//2,dim_out=dim//2, qk_scale=qk_scale, qkv_bias=qkv_bias, 
                                                            attn_drop=attn_drop, proj_drop=drop)for i in range(self.branch_num)])
        

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x, cross = True):
        """
        x: B, H*W, C
        """

        H = W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        img = self.norm1(x)
        qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3)
        
        if self.branch_num == 2:
            x1 = self.attns[0](qkv[:,:,:,:C//2], cross = False)
            x2 = self.attns[1](qkv[:,:,:,C//2:], cross = False)
            # attened_x = torch.cat([x1,x2], dim=2)
            # qkv[0,:,:,C//2:] = q1
            # qkv[0,:,:,:C//2] = q2
            # x3 = self.attns[2](qkv[:,:,:,:C//2], cross = True)
            # x4 = self.attns[3](qkv[:,:,:,C//2:], cross = True)
            x3 = self.crossattns[0](x1, x2)
            x4 = self.crossattns[1](x2, x1)
            attened_x = torch.cat([x1,x2,x3,x4], dim=2)
        else:
            attened_x = self.attns[0](qkv)
        attened_x = self.proj(attened_x)
        x = x + self.drop_path(attened_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

if __name__ == '__main__':
    x = torch.rand([1, 128 * 128, 32])
    model = CSWinBlock(32, 128, 4, 4)
    y1 = model(x, cross = True)
    y2 = model(x, cross = True)
    # z = model(x, cross = True)
    print((y1 - y2).sum())
    # x1 = img2windows(x, 1, 128)
    # x1 = x1.reshape(-1, 128, 4, 8).permute(0, 2, 1, 3).contiguous()
    # x2 = img2windows(x, 128, 1)
    # x2 = x2.reshape(-1, 128, 4, 8).permute(0, 2, 1, 3).contiguous()
    # print(x1 == x2)
    # print(x1.shape, x2.shape)
    # x1 = windows2img(x1, 1, 128, 128, 128)
    # x2 = windows2img(x2, 128, 1, 128, 128)
    # print(x1 == x2)