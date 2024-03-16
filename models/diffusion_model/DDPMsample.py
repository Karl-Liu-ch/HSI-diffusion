# jupyter notebook --no-browser --port 8890 --ip=0.0.0.0
# local
# ssh -N -f -L localhost:8890:n-62-12-19:8890 s212645@login1.hpc.dtu.dk

from IPython.display import Image
from torch.optim import Adam
import os
import sys
sys.path.append('./')
from dataset.datasets import TrainDataset, ValidDataset, TestDataset
import math
from inspect import isfunction
from functools import partial
from pathlib import Path
# %matplotlib inline
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from options import opt
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F
from utils import AverageMeter, record_loss, Loss_MRAE, Loss_RMSE, Loss_PSNR, Loss_Fid, Loss_SAM, Loss_SSIM, reconRGB, Loss_SID, SAM, SaveSpectral
import scipy.io

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
if opt.multigpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path = '/work3/s212645/Spectral_Reconstruction/checkpoint/DDPM/'

criterion_mrae = Loss_MRAE()
criterion_rmse = Loss_RMSE()
criterion_psnr = Loss_PSNR()
criterion_psnrrgb = Loss_PSNR()
criterion_sam = Loss_SAM()
criterion_sid = Loss_SID()
criterion_fid = Loss_Fid().cuda()
criterion_ssim = Loss_SSIM().cuda()

def normalize(input):
    return (input - input.min()) / (input.max() - input.min())

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    # No More Strided Convolutions or Pooling
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1),
    )


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) / (var + eps).rsqrt()

        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), 
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        self_condition=False,
        resnet_block_groups=4,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels + 3

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 1, padding=0) # changed to 1 and 0 from 7,3

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        time_dim = dim * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.out_dim = default(out_dim, channels)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


timesteps = 1000

# define beta schedule
# betas = linear_beta_schedule(timesteps=timesteps)
betas = cosine_beta_schedule(timesteps=timesteps)

# define alphas 
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


# forward diffusion (using the nice property)
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


def get_noisy_image(x_start, t):
  # add noise
  x_noisy = q_sample(x_start, t=t)

  # turn back into PIL image
  noisy_image = x_noisy[0].cpu().numpy().transpose(1,2,0)

  return noisy_image


# take time step
t = torch.tensor([10])

def p_losses(denoise_model, x_start, x_condition, t, noise=None, loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t, x_condition)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
        loss += SAM(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss

@torch.no_grad()
def p_sample(model, condition, x, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
    
    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t, condition) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise 

# Algorithm 2 (including returning all images)
@torch.no_grad()
def p_sample_loop(model, condition, shape):
    device = next(model.parameters()).device
    b = shape[0]
    img = torch.randn(shape, device=device)
    for i in reversed(range(0, timesteps)):
        img = p_sample(model, condition, img, torch.full((b,), i, device=device, dtype=torch.long), i)
    return img

@torch.no_grad()
def sample(model, condition, image_size, batch_size=16, channels=31):
    return p_sample_loop(model, condition, shape=(batch_size, channels, image_size, image_size))

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def test(model, test_loader, image_size = 128, channels = 31):
    MRAE = []
    for step, (images, labels) in enumerate(test_loader):
        labels = labels.to(device)
        images = images.to(device)
        images = Variable(images)
        labels = Variable(labels)
        samples = sample(model, images, image_size=image_size, batch_size=images.shape[0], channels=channels)
        MRAE.append(F.l1_loss(samples, labels).cpu().numpy())
    MRAE = np.array(MRAE).mean()
    return MRAE

def validate(model, valid_loader, image_size = 128, channels = 31, t_index = 1000):
    MRAE = []
    for step, (images, labels) in enumerate(valid_loader):
        labels = labels.to(device)
        images = images.to(device)
        images = Variable(images)
        labels = Variable(labels)
        x = torch.randn(labels.shape, device=device)
        b = labels.shape[0]
        samples = p_sample(model, images, x, torch.full((b,), 0, device=device, dtype=torch.long), t_index)
        MRAE.append(F.l1_loss(samples, labels).cpu().numpy())
    MRAE = np.array(MRAE).mean()
    return MRAE

results_folder = Path("./results")
results_folder.mkdir(exist_ok = True)
save_and_sample_every = 1000

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Unet(
    dim=64,
    init_dim=64,
    out_dim=31,
    channels=31,
    self_condition=True,
    dim_mults=(1, 2, 4,)
)
if opt.multigpu:
    model = nn.DataParallel(model)
model.to(device)

def save_checkpoint(epoch, model, optim, path, opt, best = False):
    if opt.multigpu:
        state = {
            'epoch': epoch,
            'model': model.module.state_dict(),
            'optim': optim.state_dict(),
        }
    else:
        state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optim': optim.state_dict(),
        }
    if best: 
        name = 'net_epoch_best.pth'
        torch.save(state, os.path.join(path, name))
    name = 'net.pth'
    torch.save(state, os.path.join(path, name))
    
def load_checkpoint(path, model, optim, epoch, opt):
    checkpoint = torch.load(os.path.join(path, 'net.pth'))
    if opt.multigpu:
        model.module.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint['model'])
    optim.load_state_dict(checkpoint['optim'])
    epoch = checkpoint['epoch']
    # try:
    #     load_metrics()
    # except:
    #     pass
    print("pretrained model loaded, iteration: ", epoch)
    return model, optim, epoch
epoch = 0
image_size = 128
channels = 31
epochs = 1000
optimizer = Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-6)
try:
    model, optimizer, epoch = load_checkpoint(path, model, optimizer, epoch, opt)
except:
    pass 

val_data = ValidDataset(data_root='/work3/s212645/Spectral_Reconstruction/', crop_size=128, valid_ratio = 0.1, test_ratio=0.1)
print("Validation set samples: ", len(val_data))
val_loader = DataLoader(dataset=val_data, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
mrae = validate(model, val_loader)
print(mrae)
train_data = TrainDataset(data_root='/work3/s212645/Spectral_Reconstruction/', crop_size=128, valid_ratio = 0.1, test_ratio=0.1)
train_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True, num_workers=2, pin_memory=True)

while epoch < epochs:
    losses = []
    for step, (images, labels) in enumerate(train_loader):
        labels = labels.to(device)
        images = images.to(device)
        images = Variable(images)
        labels = Variable(labels)
        optimizer.zero_grad()
        batch_size = labels.shape[0]
        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        t = torch.randint(0, timesteps, (batch_size,), device=device).long()
        loss = p_losses(model, labels, images, t, loss_type="huber")
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    print("Epoch: ", epoch, "Loss:", np.array(losses).mean(), 'lr: ', optimizer.param_groups[0]['lr'])
    if epoch % 10 == 0:
        mrae = validate(model, val_loader)
        print(mrae)
    save_checkpoint(epoch, model, optimizer, path, opt, best = False)
    epoch += 1
    scheduler.step()

modelname = 'DDPM'

def test(Model, modelname, batch_size = 8):
    root = '/work3/s212645/Spectral_Reconstruction/RealHyperSpectrum/' + modelname + '/'
    try: 
        os.mkdir('/work3/s212645/Spectral_Reconstruction/FakeHyperSpectrum/' + modelname + '/')
        os.mkdir('/work3/s212645/Spectral_Reconstruction/RealHyperSpectrum/' + modelname + '/')
    except:
        pass
    test_data = TestDataset(data_root=opt.data_root, crop_size=opt.patch_size, valid_ratio = 0.1, test_ratio=0.1)
    print("Test set samples: ", len(test_data))
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    Model.eval()
    losses_mrae = AverageMeter()
    losses_rmse = AverageMeter()
    losses_psnr = AverageMeter()
    losses_psnrrgb = AverageMeter()
    losses_sam = AverageMeter()
    losses_sid = AverageMeter()
    losses_fid = AverageMeter()
    losses_ssim = AverageMeter()
    for i, (input, target) in enumerate(test_loader):
        input = input.cuda()
        target = target.cuda()
        with torch.no_grad():
            # compute output
            output = sample(Model, input, image_size=128, batch_size=input.shape[0], channels=31)
            rgbs = []
            reals = []
            for j in range(output.shape[0]):
                mat = {}
                mat['cube'] = np.transpose(target[j,:,:,:].cpu().numpy(), [1,2,0])
                mat['rgb'] = np.transpose(input[j,:,:,:].cpu().numpy(), [1,2,0])
                real = mat['rgb']
                real = (real - real.min()) / (real.max()-real.min())
                mat['rgb'] = real
                scipy.io.savemat(root + str(i * output.shape[0] + j).zfill(3) + '.mat', mat)
                rgb = SaveSpectral(output[j,:,:,:], i * output.shape[0] + j, root='/work3/s212645/Spectral_Reconstruction/FakeHyperSpectrum/' + modelname + '/')
                rgbs.append(rgb)
                reals.append(real)
                print(i * output.shape[0] + j, 'saved')
            loss_mrae = criterion_mrae(output, target)
            loss_rmse = criterion_rmse(output, target)
            loss_psnr = criterion_psnr(output, target)
            loss_sam = criterion_sam(output, target)
            loss_sid = criterion_sid(output, target)
            rgbs = np.array(rgbs).transpose(0, 3, 1, 2)
            rgbs = torch.from_numpy(rgbs).cuda()
            reals = np.array(reals).transpose(0, 3, 1, 2)
            reals = torch.from_numpy(reals).cuda()
            loss_fid = criterion_fid(rgbs, reals)
            loss_ssim = criterion_ssim(rgbs, reals)
            loss_psrnrgb = criterion_psnrrgb(rgbs, reals)
        # record loss
        losses_mrae.update(loss_mrae.data)
        losses_rmse.update(loss_rmse.data)
        losses_psnr.update(loss_psnr.data)
        losses_sam.update(loss_sam.data)
        losses_sid.update(loss_sid.data)
        losses_fid.update(loss_fid.data)
        losses_ssim.update(loss_ssim.data)
        losses_psnrrgb.update(loss_psrnrgb.data)
    criterion_sam.reset()
    criterion_psnr.reset()
    criterion_psnrrgb.reset()
    return losses_mrae.avg, losses_rmse.avg, losses_psnr.avg, losses_sam.avg, losses_sid.avg, losses_fid.avg, losses_ssim.avg, losses_psnrrgb.avg


file = '/zhome/02/b/164706/Master_Courses/2023_Fall/Spectral_Reconstruction/result.txt'
f = open(file, 'a')
mrad, rmse, psnr, sam, sid, fid, ssim, psnrrgb = test(model, modelname, batch_size=8)
print(f'MRAE:{mrad}, RMSE: {rmse}, PNSR:{psnr}, SAM: {sam}, SID: {sid}, FID: {fid}, SSIM: {ssim}, PSNRRGB: {psnrrgb}')
f.write(modelname+':\n')
f.write(f'MRAE:{mrad}, RMSE: {rmse}, PNSR:{psnr}, SAM: {sam}, SID: {sid}, FID: {fid}, SSIM: {ssim}, PSNRRGB: {psnrrgb}')
f.write('\n')
f.close()