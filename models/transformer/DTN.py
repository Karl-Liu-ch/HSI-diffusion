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
from models.transformer.sn_swin_transformer_v2 import SwinTransformerBlock as SN_SwinTransformerBlock
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

class SWTB(nn.Module):
    def __init__(self, dim, input_resolution, num_heads=1, window_size=8, shift_size=0):
        super().__init__()
        self.input_resolution = input_resolution
        self.model = nn.Sequential(
            SwinTransformerBlock_v2(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size, shift_size=0),
            SwinTransformerBlock_v2(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size, shift_size=window_size // 2)
        )
        # self.model = SwinTransformerBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size)
        # self.model = SwinTransformerBlock_v2(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size, shift_size=shift_size)
        # self.model = AgentSwin(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size)
        # agent swin for later
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
                 shift_size=0):
        super().__init__()
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
        spatialinter = torch.sigmoid(self.spatial_interaction(convx))
        
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
                 heads=1):
        super().__init__()
        self.model = MSAB(dim=dim, num_blocks=num_blocks, dim_head=dim_head, heads=heads, use_conv=False)
        
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
        self.ffn = SGFN(in_features=dim, hidden_features=dim*2)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
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
        pool = torch.concat([self.avgpool(convx), self.maxpool(convx)], dim = 1)

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
    def __init__(self, dim, dim_head, input_resolution, num_heads, window_size, num_block, num_msab = 2):
        super().__init__()
        self.dim = dim
        self.dim_head = dim_head
        self.input_resolution = tuple(input_resolution)
        self.num_heads = num_heads
        self.window_size = window_size
        layer = []
        for i in range(num_block):
            layer += [Adaptive_MSAB(dim, num_blocks=num_msab, dim_head=dim_head, heads=dim // dim_head)]
            layer += [Adaptive_SWTB(dim, self.input_resolution, num_heads=dim // dim_head, window_size=window_size)]
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
        self.model = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=4, stride=2, padding=1, bias=False)
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
        self.model = nn.Sequential(
            nn.ConvTranspose2d(inchannel, outchannel, stride=2, kernel_size=2, padding=0, output_padding=0),
        )
        # self.model = nn.Sequential(
        #     nn.Conv2d(inchannel, inchannel * 4, kernel_size=1, stride=1, padding=0, bias=False), 
        #     nn.GELU(), 
        #     nn.PixelShuffle(2),
        #     nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False), 
        # )
        
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
                 num_msab = 2):
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
                         num_msab=num_msab),
                DownSample(dim_stage, dim_stage * 2),
            ]))
            img_size[0] = img_size[0] // 2
            img_size[1] = img_size[1] // 2
            dim_stage *= 2
        
        self.bottleneck = DTNBlock(dim = dim_stage, 
                         dim_head = dim, 
                         input_resolution = img_size, 
                         num_heads = dim_stage // dim, 
                         window_size = window_size, 
                         num_block = bottleblock)
        
        # Decoder
        self.decoder_layers = nn.ModuleList([])
        n_block.reverse()
        for num_block in n_block:
            img_size[0] = img_size[0] * 2
            img_size[1] = img_size[1] * 2
            self.decoder_layers.append(nn.ModuleList([
                UpSample(dim_stage, dim_stage // 2), 
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),
                DTNBlock(dim = dim_stage // 2, 
                         dim_head = dim, 
                         input_resolution = img_size, 
                         num_heads = (dim_stage // 2) // dim, 
                         window_size = window_size, 
                         num_block = num_block),
            ]))
            dim_stage //= 2
        
        self.mapping = nn.Conv2d(dim, out_dim, 3, 1, 1, bias=False)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        # elif isinstance(m, nn.LayerNorm):
        #     nn.init.constant_(m.bias, 0)
        #     nn.init.constant_(m.weight, 1.0)

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

class TrainDTN(BaseModel):
    def __init__(self, opt, model, model_name, multiGPU=False) -> None:
        super().__init__(opt, model, model_name, multiGPU)
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=opt.init_lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.end_epoch, eta_min=1e-6)    

    def test_full_resol(self):
        modelname = self.name
        try:
            os.mkdir('/work3/s212645/Spectral_Reconstruction/FakeHyperSpectrum/' + modelname + '/FullResol/')
        except:
            pass
        root = '/work3/s212645/Spectral_Reconstruction/RealHyperSpectrum/'
        H_ = 128
        W_ = 128
        losses_mrae = AverageMeter()
        losses_rmse = AverageMeter()
        losses_psnr = AverageMeter()
        losses_psnrrgb = AverageMeter()
        losses_sam = AverageMeter()
        losses_sid = AverageMeter()
        losses_fid = AverageMeter()
        losses_ssim = AverageMeter()
        test_data = TestFullDataset(data_root=opt.data_root, crop_size=opt.patch_size, valid_ratio = 0.1, test_ratio=0.1)
        print("Test set samples: ", len(test_data))
        test_loader = DataLoader(dataset=test_data, batch_size=opt.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        count = 0
        for i, (input, target) in enumerate(test_loader):
            input = input.cuda()
            target = target.cuda()
            H, W = input.shape[-2], input.shape[-1]
            if H != H_ or W != W_:
                self.G = DTN(in_dim=3, 
                        out_dim=31,
                        img_size=[H, W], 
                        window_size=8, 
                        n_block=[2,2,2,2], 
                        bottleblock = 4).to(device)
                H_ = H
                W_ = W
                self.load_checkpoint()
            with torch.no_grad():
                output = self.G(input)
                rgbs = []
                reals = []
                for j in range(output.shape[0]):
                    mat = {}
                    mat['cube'] = np.transpose(target[j,:,:,:].cpu().numpy(), [1,2,0])
                    mat['rgb'] = np.transpose(input[j,:,:,:].cpu().numpy(), [1,2,0])
                    real = mat['rgb']
                    real = (real - real.min()) / (real.max()-real.min())
                    mat['rgb'] = real
                    rgb = SaveSpectral(output[j,:,:,:], count, root='/work3/s212645/Spectral_Reconstruction/FakeHyperSpectrum/' + modelname + '/FullResol/')
                    rgbs.append(rgb)
                    reals.append(real)
                    count += 1
                loss_mrae = criterion_mrae(output, target)
                loss_rmse = criterion_rmse(output, target)
                loss_psnr = criterion_psnr(output, target)
                loss_sam = criterion_sam(output, target)
                loss_sid = criterion_sid(output, target)
                rgbs = np.array(rgbs).transpose(0, 3, 1, 2)
                rgbs = torch.from_numpy(rgbs).cuda()
                reals = np.array(reals).transpose(0, 3, 1, 2)
                reals = torch.from_numpy(reals).cuda()
                # loss_fid = criterion_fid(rgbs, reals)
                loss_ssim = criterion_ssim(rgbs, reals)
                loss_psrnrgb = criterion_psnrrgb(rgbs, reals)
            # record loss
            losses_mrae.update(loss_mrae.data)
            losses_rmse.update(loss_rmse.data)
            losses_psnr.update(loss_psnr.data)
            losses_sam.update(loss_sam.data)
            losses_sid.update(loss_sid.data)
            # losses_fid.update(loss_fid.data)
            losses_ssim.update(loss_ssim.data)
            losses_psnrrgb.update(loss_psrnrgb.data)
        criterion_sam.reset()
        criterion_psnr.reset()
        criterion_psnrrgb.reset()
        file = '/zhome/02/b/164706/Master_Courses/thesis/HSI-diffusion/result.txt'
        f = open(file, 'a')
        f.write(modelname+':\n')
        f.write(f'MRAE:{losses_mrae.avg}, RMSE: {losses_rmse.avg}, PNSR:{losses_psnr.avg}, SAM: {losses_sam.avg}, SID: {losses_sid.avg}, SSIM: {losses_ssim.avg}, PSNRRGB: {losses_psnrrgb.avg}')
        f.write('\n')
        return losses_mrae.avg, losses_rmse.avg, losses_psnr.avg, losses_sam.avg, losses_sid.avg, losses_fid.avg, losses_ssim.avg, losses_psnrrgb.avg


class DTN_multi_stage(nn.Module):
    def __init__(self, *, in_channels=3, out_channels=31, n_feat=31, stage=3, img_size=[128, 128], window = 32, num_msab = 1, **kargs):
        super(DTN_multi_stage, self).__init__()
        self.stage = stage
        self.conv_in = nn.Conv2d(in_channels, n_feat, kernel_size=3, padding=(3 - 1) // 2,bias=False)
        self.hb, self.wb = window, window
        pad_h = (self.hb - img_size[0] % self.hb) % self.hb
        pad_w = (self.wb - img_size[1] % self.wb) % self.wb
        modules_body = [DTN(in_dim=n_feat, 
                            out_dim=out_channels,
                            img_size = [img_size[0]+pad_h, img_size[1]+pad_w], 
                            window_size = 8, 
                            n_block=[1,1], 
                            bottleblock = 1,
                            num_msab = num_msab) for _ in range(stage)]
        self.body = nn.Sequential(*modules_body)
        self.conv_out = nn.Conv2d(n_feat * 2, out_channels, kernel_size=3, padding=(3 - 1) // 2,bias=False)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        b, c, h_inp, w_inp = x.shape
        pad_h = (self.hb - h_inp % self.hb) % self.hb
        pad_w = (self.wb - w_inp % self.wb) % self.wb
        # x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='constant', value=0.0)
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
                    n_block=[2,2,2,2], 
                    bottleblock = 4).to(device)
    # model.apply(init_weights_uniform)
    # model = DTN_multi_stage(in_channels=3, out_channels=31, n_feat=31, img_size=[128, 128], window=32).to(device)
    input = torch.rand([1, 3, 128, 128]).to(device)
    output = model(input.float())
    print(output.shape)
    summary(model, (3, 128, 128))
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