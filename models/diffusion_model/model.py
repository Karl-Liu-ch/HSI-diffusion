import sys
sys.path.append('./')
import re
# %matplotlib inline
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from inspect import isfunction
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from dataset.datasets import TrainDataset, ValidDataset, TestDataset, TestFullDataset
from torch.optim import Adam
from pathlib import Path
from torch.utils.data import Dataset
import torch
from torch import nn, einsum
import torch.nn.functional as F
from options import opt

from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import os
from Models.Diffusion_Model.networks import Unet
import platform
from utils import * 
from visualize import *

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
if opt.multigpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

criterion_mrae = Loss_MRAE()
criterion_rmse = Loss_RMSE()
criterion_psnr = Loss_PSNR()
criterion_psnrrgb = Loss_PSNR()
criterion_sam = Loss_SAM()
criterion_sid = Loss_SID()
criterion_fid = Loss_Fid().cuda()
criterion_ssim = Loss_SSIM().cuda()

def transform(tensor):
    tensor = tensor.clone().detach()
    return tensor * 2.0 - 1.0
def reverse_transform(tensor):
    tensor = tensor.clone().detach()
    return (tensor+1.0) / 2.0

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    betas = betas.astype(np.float32)
    return torch.from_numpy(betas).to(device)

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

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

class Diffusion():
    def __init__(self, betas, timesteps = 1000, img_size = [128, 128]):
        self.model = Unet(
        64,
        init_dim = 64,
        out_dim = 31,
        dim_mults = (1, 2, 4, 8),
        channels = 31,
        cond_channels=6,
        self_condition = True,
        resnet_block_groups = 8,
        learned_variance = True,
        learned_sinusoidal_cond = True,
        random_fourier_features = True,
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000,
        attn_dim_head = 32,
        attn_heads = 4,
        full_attn = None,    # defaults to full attention only for inner most layer
        flash_attn = False)
        if opt.multigpu:
            self.model = nn.DataParallel(self.model)
        self.model.to(device)
        
        self.channels = 31
        self.epochs = 200
        self.epoch = 0
        self.optimizer = Adam(self.model.parameters(), lr=4e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.epochs, eta_min=1e-6)
        
        if platform.system().lower() == 'linux':
            self.path = '/work3/s212645/Spectral_Reconstruction/checkpoint/DDPM/'
        elif platform.system().lower() == 'windows':
            self.path = 'H:/深度学习/checkpoint/DDPM/'
        
        self.betas = betas
        self.timesteps = timesteps
        self.alphas = 1. - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.img_size = img_size
            
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start).to(device)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def get_noisy_image(self, x_start, t):
        # add noise
        x_noisy = self.q_sample(x_start, t=t)

        # turn back into PIL image
        noisy_image = x_noisy[0].cpu().numpy().transpose(1,2,0)

        return noisy_image

    def p_losses(self, x_start, t, noise=None, loss_type="l1", cond=None):
        if noise is None:
            noise = torch.randn_like(x_start).to(device)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.model(x_noisy, t, cond)

        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            loss = F.l1_loss(noise, predicted_noise) + F.mse_loss(noise, predicted_noise)

        return loss
    
    @torch.no_grad()
    def p_sample(self, x, t, t_index, cond = None):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)
        
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.model(x, t, cond) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise 

    # Algorithm 2 (including returning all images)
    @torch.no_grad()
    def p_sample_loop(self, shape, noise=None, cond = None):
        device = next(self.model.parameters()).device

        b = shape[0]
        # start from pure noise (for each example in the batch)
        if noise is None:
            img = torch.randn(shape, device=device)
        else:
            img = noise

        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), i, cond = cond)
        return img

    @torch.no_grad()
    def sample(self, image_size=[128, 128], batch_size=16, channels=31, noise=None, cond = None):
        return self.p_sample_loop(shape=(batch_size, channels, image_size[0], image_size[1]), noise = noise, cond = cond)
    
    def save_checkpoint(self):
        state = {
                'epoch': self.epoch,
                'model': self.model.state_dict(),
                'optim': self.optimizer.state_dict(),
            }
        name = 'net.pth'
        torch.save(state, os.path.join(self.path, name))
        
    def load_checkpoint(self):
        checkpoint = torch.load(os.path.join(self.path, 'net.pth'))
        self.epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optim'])
        print("pretrained model loaded, iteration: ", self.epoch)
    
    def train(self, train_loader, val_loader):
        while self.epoch < self.epochs:
            losses = []
            for step, (images, labels, rgb_gt) in enumerate(train_loader):
                images = transform(images)
                labels = transform(labels)
                labels = labels.to(device)
                images = images.to(device)
                images = Variable(images)
                labels = Variable(labels)
                self.optimizer.zero_grad()
                batch_size = labels.shape[0]
                # Algorithm 1 line 3: sample t uniformally for every example in the batch
                t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()
                loss = self.p_losses(labels, t, loss_type="huber", cond = images)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
            print("Epoch: ", self.epoch, "Loss:", np.array(losses).mean(), 'lr: ', self.optimizer.param_groups[0]['lr'])
            if self.epoch % 10 == 0:
                mrae = self.validate(val_loader)
                print(mrae)
            self.save_checkpoint()
            self.epoch += 1
            self.scheduler.step()
    
    def test(self, test_loader, modelname='DDPM'):
        try: 
            os.mkdir('/work3/s212645/Spectral_Reconstruction/FakeHyperSpectrum/')
            os.mkdir('/work3/s212645/Spectral_Reconstruction/RealHyperSpectrum/')
        except:
            pass
        try: 
            os.mkdir('/work3/s212645/Spectral_Reconstruction/FakeHyperSpectrum/' + modelname + '/')
            os.mkdir('/work3/s212645/Spectral_Reconstruction/RealHyperSpectrum/' + modelname + '/')
        except:
            pass
        self.model.eval()
        losses_mrae = AverageMeter()
        losses_rmse = AverageMeter()
        losses_psnr = AverageMeter()
        losses_psnrrgb = AverageMeter()
        losses_sam = AverageMeter()
        losses_sid = AverageMeter()
        losses_fid = AverageMeter()
        losses_ssim = AverageMeter()
        for i, (images, labels, rgb_gt) in enumerate(test_loader):
            images = transform(images)
            labels = transform(labels)
            images = images.cuda()
            labels = labels.cuda()
            with torch.no_grad():
                # compute output
                output = self.sample(image_size=self.img_size, batch_size=images.shape[0], channels=31, cond = images)
                output = reverse_transform(output)
                labels = reverse_transform(labels)
                images = reverse_transform(images)
                rgbs = []
                reals = []
                for j in range(output.shape[0]):
                    mat = {}
                    mat['cube'] = np.transpose(labels[j,:,:,:].cpu().numpy(), [1,2,0])
                    mat['rgb'] = np.transpose(rgb_gt[j,:,:,:].cpu().numpy(), [1,2,0])
                    real = mat['rgb']
                    real = (real - real.min()) / (real.max()-real.min())
                    mat['rgb'] = real
                    rgb = SaveSpectral(output[j,:,:,:], i * opt.batch_size + j, root='/work3/s212645/Spectral_Reconstruction/FakeHyperSpectrum/' + modelname + '/')
                    rgbs.append(rgb)
                    reals.append(real)
                loss_mrae = criterion_mrae(output, labels)
                loss_rmse = criterion_rmse(output, labels)
                loss_psnr = criterion_psnr(output, labels)
                loss_sam = criterion_sam(output, labels)
                loss_sid = criterion_sid(output, labels)
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
        file = '/zhome/02/b/164706/Master_Courses/2023_Fall/Spectral_Reconstruction/result.txt'
        f = open(file, 'a')
        f.write(modelname+':\n')
        f.write(f'MRAE:{losses_mrae.avg}, RMSE: {losses_rmse.avg}, PNSR:{losses_psnr.avg}, SAM: {losses_sam.avg}, SID: {losses_sid.avg}, FID: {losses_fid.avg}, SSIM: {losses_ssim.avg}, PSNRRGB: {losses_psnrrgb.avg}')
        f.write('\n')
        return losses_mrae.avg, losses_rmse.avg, losses_psnr.avg, losses_sam.avg, losses_sid.avg, losses_fid.avg, losses_ssim.avg, losses_psnrrgb.avg

    def validate(self, valid_loader, image_size = 128, channels = 31, t_index = 1000):
        MRAE = []
        for step, (images, labels, rgb_gt) in enumerate(valid_loader):
            labels = labels.to(device)
            images = images.to(device)
            images = Variable(images)
            labels = Variable(labels)
            b, c, h, w = labels.shape
            samples = self.sample(image_size=[h, w], batch_size=b, channels=31, cond = images)
            MRAE.append(F.l1_loss(samples, labels).cpu().numpy())
            break
        MRAE = np.array(MRAE).mean()
        return MRAE
    
    def test_full(self, test_loader, modelname='DDPM'):
        try: 
            os.mkdir('/work3/s212645/Spectral_Reconstruction/FakeHyperSpectrum/')
            os.mkdir('/work3/s212645/Spectral_Reconstruction/RealHyperSpectrum/')
        except:
            pass
        try: 
            os.mkdir('/work3/s212645/Spectral_Reconstruction/FakeHyperSpectrum/' + modelname + '/')
            os.mkdir('/work3/s212645/Spectral_Reconstruction/RealHyperSpectrum/' + modelname + '/')
        except:
            pass
        self.model.eval()
        losses_mrae = AverageMeter()
        losses_rmse = AverageMeter()
        losses_psnr = AverageMeter()
        losses_psnrrgb = AverageMeter()
        losses_sam = AverageMeter()
        losses_sid = AverageMeter()
        losses_fid = AverageMeter()
        losses_ssim = AverageMeter()
        for i, (images, labels, rgb_gt) in enumerate(test_loader):
            images = transform(images)
            labels = transform(labels)
            images = images.cuda()
            labels = labels.cuda()
            b, c, h, w = labels.shape
            with torch.no_grad():
                # compute output
                output = self.sample(image_size=[h, w], batch_size=b, channels=31, cond = images)
                output = reverse_transform(output)
                labels = reverse_transform(labels)
                images = reverse_transform(images)
                rgbs = []
                reals = []
                for j in range(output.shape[0]):
                    mat = {}
                    mat['cube'] = np.transpose(labels[j,:,:,:].cpu().numpy(), [1,2,0])
                    mat['rgb'] = np.transpose(rgb_gt[j,:,:,:].cpu().numpy(), [1,2,0])
                    real = mat['rgb']
                    real = (real - real.min()) / (real.max()-real.min())
                    mat['rgb'] = real
                    rgb = SaveSpectral(output[j,:,:,:], i * opt.batch_size + j, root='/work3/s212645/Spectral_Reconstruction/FakeHyperSpectrum/' + modelname + '/')
                    rgbs.append(rgb)
                    reals.append(real)
                loss_mrae = criterion_mrae(output, labels)
                loss_rmse = criterion_rmse(output, labels)
                loss_psnr = criterion_psnr(output, labels)
                loss_sam = criterion_sam(output, labels)
                loss_sid = criterion_sid(output, labels)
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
        file = '/zhome/02/b/164706/Master_Courses/2023_Fall/Spectral_Reconstruction/result.txt'
        f = open(file, 'a')
        f.write(modelname+':\n')
        f.write(f'MRAE:{losses_mrae.avg}, RMSE: {losses_rmse.avg}, PNSR:{losses_psnr.avg}, SAM: {losses_sam.avg}, SID: {losses_sid.avg}, FID: {losses_fid.avg}, SSIM: {losses_ssim.avg}, PSNRRGB: {losses_psnrrgb.avg}')
        f.write('\n')
        return losses_mrae.avg, losses_rmse.avg, losses_psnr.avg, losses_sam.avg, losses_sid.avg, losses_fid.avg, losses_ssim.avg, losses_psnrrgb.avg

    
if __name__ == '__main__':
    timesteps = 1000
    # define beta schedule
    # betas = linear_beta_schedule(timesteps=timesteps)
    betas = get_beta_schedule('sigmoid', beta_start=0.00001, beta_end=0.02, num_diffusion_timesteps=timesteps)
    # betas = torch.from_numpy(betas)

    Diff = Diffusion(betas=betas, timesteps=timesteps)

    if opt.loadmodel:
        try:
            Diff.load_checkpoint()
            print('model loaded')
        except:
            print('model loading failed')
    
    if opt.mode == 'train':
        train_data = TrainDataset(data_root='/work3/s212645/Spectral_Reconstruction/', crop_size=opt.patch_size, valid_ratio = 0.1, test_ratio=0.1, datanames = ['ARAD/'])
        train_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_data = ValidDataset(data_root='/work3/s212645/Spectral_Reconstruction/', crop_size=opt.patch_size, valid_ratio = 0.1, test_ratio=0.1, datanames = ['ARAD/'])
        val_loader = DataLoader(dataset=val_data, batch_size=opt.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        Diff.train(train_loader, val_loader)
    
    elif opt.mode == 'test':
        Diff.load_checkpoint()
        test_data = TestDataset(data_root='/work3/s212645/Spectral_Reconstruction/', crop_size=opt.patch_size, valid_ratio = 0.1, test_ratio=0.1, datanames = ['ARAD/'])
        test_loader = DataLoader(dataset=test_data, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
        Diff.test(test_loader)

    elif opt.mode == 'sample':
        Diff.load_checkpoint()
        test_data = TestFullDataset(data_root=opt.data_root, crop_size=opt.patch_size, valid_ratio = 0.1, test_ratio=0.1, datanames=['ARAD/'])
        test_loader = DataLoader(dataset=test_data, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
        Diff.test_full(test_loader)
        for path, dir, files in os.walk('results/diffusion/'):
            for file in files:
                os.remove(f'{path}/{file}')
        form = '.png'
        form = re.compile(form)
        (images, labels, rgb_gt) = next(iter(test_loader))
        labels = labels * 2.0 - 1.0
        images = images * 2.0 - 1.0
        print(labels[0].dtype)
        x_start = labels.to(device)
        t_ = [0, 10, 100, 200, 500, 999]
        visualize_tensor((x_start + 1.0) / 2.0, 'gt', 'diffusion')
        for t in t_:
            tt = torch.tensor([t]).to(device)
            x_noisy = Diff.q_sample(x_start, t=tt)
            x_noisy = (x_noisy + 1.0) / 2.0
            visualize_tensor(x_noisy, t, 'diffusion')
        images = []
        for path, dir, files in os.walk('results/diffusion/'):
            files.sort()
            for file in files:
                if form.search(file) is not None:
                    img = Image.open(f'{path}/{file}')
                    images.append(img)
        idx = len(images)
        gap_height = 0
        result_width = img.width * idx + gap_height * (idx - 1)
        result_image = Image.new('RGB', (result_width, img.height), color='white')
        i = 0
        for img in images:
            result_image.paste(img, ((img.width + gap_height) * i, 0))
            i += 1
        result_image.save(f'results/diffusion/concat.png', format='PNG', compress_level=0)