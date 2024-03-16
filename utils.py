from __future__ import division

import torch
import torch.nn as nn
import logging
import numpy as np
import os
from torch import nn, optim
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import SpectralAngleMapper, PeakSignalNoiseRatio
from torchmetrics.image.fid import FrechetInceptionDistance
from spectral import *
import cv2
from NTIRE2022Util import *
import scipy.io
import cv2
from einops import repeat
import importlib
rgbfilterpath = 'resources/RGB_Camera_QE.csv'
camera_filter, filterbands = load_rgb_filter(rgbfilterpath)

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def normalization_image(x):
    B, C, H, W = x.shape
    v_min = x.view(B, -1).min(dim=1)[0]
    v_max = x.view(B, -1).max(dim=1)[0]
    v_min = repeat(v_min, 'b -> b c h w', c = C, h = H, w = W)
    v_max = repeat(v_max, 'b -> b c h w', c = C, h = H, w = W)
    x_norm = (x - v_min) / (v_max - v_min)
    return x_norm

def Normalize(x):
    return (x - x.min()) / (x.max() - x.min())

def Image256(x):
    image = np.round(x * 255.0)
    return image.astype(np.uint8)

def RGB2YCrCb(rgb_image):
    yuv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2YCrCb)
    return yuv_image

def YUV2RGB(yuv_image):
    rgb_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2RGB)
    return rgb_image

def bgr2rgb(bgr):
    bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    bgr = np.float32(bgr)
    rgb = (bgr-bgr.min())/(bgr.max()-bgr.min())
    return rgb

def reconRGB(labels):
    cube_bands = np.linspace(400,700,31)
    b = labels.shape[0]
    labels = labels.cpu().numpy()
    rgbs = []
    for i in range(b):
        label = np.transpose(labels[i,:,:,:], [1,2,0])
        rgb = projectHS(label, cube_bands, camera_filter, filterbands, clipNegative=True)
        rgb = (rgb-rgb.min())/(rgb.max()-rgb.min())
        rgb = np.transpose(rgb, [2,0,1])
        rgbs.append(rgb)
    rgbs = np.array(rgbs)
    rgbs = torch.from_numpy(rgbs).cuda()
    return rgbs

def reconRGBfromNumpy(labels):
    cube_bands = np.linspace(400,700,31)
    rgb = projectHS(labels, cube_bands, camera_filter, filterbands, clipNegative=True)
    rgb = (rgb-rgb.min())/(rgb.max()-rgb.min())
    return rgb

def SaveSpectral(spectensor, i, root = '/work3/s212645/Spectral_Reconstruction/FakeHyperSpectrum/'):
    mat = {}
    specnp = np.transpose(spectensor.cpu().numpy(), [1,2,0])
    name = str(i).zfill(3) + '.mat'
    mat['cube'] = specnp
    rgb = reconRGBfromNumpy(specnp)
    mat['rgb'] = rgb
    scipy.io.savemat(root + name, mat, do_compression=True)
    return rgb

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def initialize_logger(file_dir):
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename=file_dir, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(message)s', "%Y-%m-%d %H:%M:%S")
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)
    return logger

def save_checkpoint(model_path, epoch, iteration, model, optimizer):
    state = {
        'epoch': epoch,
        'iter': iteration,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    torch.save(state, os.path.join(model_path, 'net_%depoch.pth' % epoch))

class Loss_Fid(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fid = FrechetInceptionDistance(feature=2048)
        
    def forward(self, outputs, label):
        outputs = outputs * 255
        label = label * 255
        if torch.cuda.is_available():
            outputs = outputs.type(torch.cuda.ByteTensor)
            label = label.type(torch.cuda.ByteTensor)
        else:
            outputs = outputs.type(torch.ByteTensor)
            label = label.type(torch.ByteTensor)
        self.fid.update(label, real=True)
        self.fid.update(outputs, real=False)
        return self.fid.compute()
    
    def reset(self):
        self.fid.reset()

class Loss_SAM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sam = SpectralAngleMapper()
    
    def forward(self, outputs, label):
        sam_score = self.sam(outputs.cpu(), label.cpu())
        sam_score = torch.mean(sam_score.view(-1))
        return sam_score
    
    def reset(self):
        self.sam.reset()
        # print('SpectralAngleMapper reseted')
        
class SAMLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, preds, target):
        assert preds.shape == target.shape
        dot_product = (preds * target).sum(dim=1)
        preds_norm = preds.norm(dim=1)
        target_norm = target.norm(dim=1)
        sam_score = torch.clamp(dot_product / (preds_norm * target_norm), -1, 1).acos()
        return torch.mean(sam_score)
        
def SAM(preds, target):
    dot_product = (preds * target).sum(dim=1)
    preds_norm = preds.norm(dim=1)
    target_norm = target.norm(dim=1)
    sam_score = torch.clamp(dot_product / (preds_norm * target_norm), -1, 1).acos()
    return torch.mean(sam_score)

def SAMHeatMap(preds, target):
    dot_product = np.sum(preds * target, axis=2)
    preds_norm = np.linalg.norm(preds, axis=2)
    target_norm = np.linalg.norm(target, axis=2)
    sam_score = np.arccos(dot_product / (preds_norm * target_norm))
    return sam_score

class Loss_SSIM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        
    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        ssim_score = self.ssim(outputs, label)
        ssim_score = torch.mean(ssim_score.view(-1))
        return ssim_score
    
    def reset(self):
        self.ssim.reset()

class Loss_MRAE(nn.Module):
    def __init__(self):
        super(Loss_MRAE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        if label.all() == False:
            error = torch.abs(outputs - label) / (label + 1e-5)
        else:
            error = torch.abs(outputs - label) / label
        mrae = torch.mean(error.reshape(-1))
        # mrae = torch.mean(error.view(-1))
        return mrae

class Loss_RMSE(nn.Module):
    def __init__(self):
        super(Loss_RMSE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = outputs-label
        sqrt_error = torch.pow(error,2)
        rmse = torch.sqrt(torch.mean(sqrt_error.reshape(-1)))
        # rmse = torch.sqrt(torch.mean(sqrt_error.view(-1)))
        return rmse

# class Loss_PSNR(nn.Module):
#     def __init__(self):
#         super(Loss_PSNR, self).__init__()

#     def forward(self, im_true, im_fake, data_range=255):
#         N = im_true.size()[0]
#         C = im_true.size()[1]
#         H = im_true.size()[2]
#         W = im_true.size()[3]
#         Itrue = im_true.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)
#         Ifake = im_fake.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)
#         mse = nn.MSELoss(reduce=False)
#         err = mse(Itrue, Ifake).sum(dim=1, keepdim=True).div_(C * H * W)
#         psnr = 10. * torch.log((data_range ** 2) / err) / np.log(10.)
#         return torch.mean(psnr)

class Loss_PSNR(nn.Module):
    def __init__(self):
        super(Loss_PSNR, self).__init__()
        self.psnr = PeakSignalNoiseRatio()

    def forward(self, im_fake, im_true):
        psnr_score = self.psnr(im_fake.cpu(), im_true.cpu())
        psnr_score = torch.mean(psnr_score.view(-1))
        return psnr_score
    
    def reset(self):
        self.psnr.reset()
        # print('Peak Signal Noise Ratio reseted')

class Loss_SID(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, imfake, imreal):
        b = imreal.shape[0]
        imfake = imfake.reshape(b, -1)
        imreal = imreal.reshape(b, -1)
        p = (imfake / torch.sum(imfake)) + torch.finfo(torch.float).eps
        q = (imreal / torch.sum(imreal)) + torch.finfo(torch.float).eps
        return torch.mean(torch.sum(p * torch.log(p / q) + q * torch.log(q / p), dim=1).reshape(-1))

def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename

def record_loss(loss_csv, epoch, iteration, epoch_time, lr, train_loss, test_loss):
    """ Record many results."""
    loss_csv.write('{},{},{},{},{},{}\n'.format(epoch, iteration, epoch_time, lr, train_loss, test_loss))
    loss_csv.flush()
    loss_csv.close
    
if __name__ == '__main__':
    criterion_ssim = Loss_SSIM()
    criterion_sam = Loss_SAM()
    criterion_mrae = Loss_MRAE()
    criterion_rmse = Loss_RMSE()
    criterion_psnr = Loss_PSNR()
    target = torch.randn([64, 31, 128, 128]).cuda()
    pred = target * 0.75
    print(criterion_sam(pred, target))
    criterion_sam.reset()
    
    