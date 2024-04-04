import sys
sys.path.append('./')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from models.gan.networks import *
from utils import instantiate_from_config
from models.vae.networks import Encoder
from omegaconf import OmegaConf

def load_hsi_perceptual_encoder():
    ckpt_path = '/work3/s212645/Spectral_Reconstruction/checkpoint/vae_perceptual/lightning_logs/version_0/checkpoints/epoch147-mrae_avg0.03.ckpt'
    ckpt = torch.load(ckpt_path)
    encoder_weights = {k: v for k, v in ckpt["state_dict"].items() if k.startswith("encoder.")}
    keys = []
    newkeys = []
    for key in encoder_weights.keys():
        keys.append(key)
        newkey = key[8:]
        newkeys.append(newkey)
    for key, newkey in zip(keys, newkeys):
        encoder_weights[newkey] = encoder_weights.pop(key)
    cfg = OmegaConf.load('configs/hsi_vae_perceptual.yaml').model.params.ddconfig
    model = Encoder(**cfg).cuda()
    model.load_state_dict(encoder_weights)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model

class SamLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        dot_product = (input * target).sum(dim=1)
        preds_norm = input.norm(dim=1)
        target_norm = target.norm(dim=1)
        sam_score = torch.clamp(dot_product / (preds_norm * target_norm), -1, 1).acos()
        return torch.mean(sam_score)

class Wgan_Loss(nn.Module):
    def __init__(self, *, discconfig, l1_weight, sam_weight, features_weight, **kwargs) -> None:
        super().__init__()
        self.discriminator = instantiate_from_config(discconfig)
        self.l1_loss = nn.L1Loss()
        self.l1_weight = l1_weight
        self.sam_loss = SamLoss()
        self.sam_weight = sam_weight
        self.features_weight = features_weight

    def calculate_adaptive_weight(self, rec_loss, g_loss, last_layer=None):
        if last_layer is not None:
            rec_grads = torch.autograd.grad(rec_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            rec_grads = torch.autograd.grad(rec_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(rec_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        # d_weight = d_weight * self.discriminator_weight
        return d_weight
    
    def forward(self, reconstructions, labels, cond, optimizer_idx, split="train", last_layer = None):
        l1_loss = self.l1_loss(reconstructions, labels)
        sam_loss = self.sam_loss(reconstructions, labels)
        rec_loss = l1_loss * self.l1_weight + sam_loss * self.sam_weight

        if optimizer_idx == 0:
            # train generator
            # for params in self.discriminator.parameters():
            #     params.requires_grad = False
            disc_fake, fake_features = self.discriminator(torch.concat([reconstructions, cond], dim=1))
            disc_real, real_features = self.discriminator(torch.concat([labels, cond], dim=1))
            disc_fake *= self.calculate_adaptive_weight(rec_loss, disc_fake, last_layer)

            feature_loss = 0
            for k in range(len(fake_features)):
                feature_loss += F.mse_loss(real_features[k].detach(), fake_features[k])
            total_loss = - disc_fake.mean(0).view(1) + rec_loss + feature_loss * self.features_weight
            log = {"{}/total_loss".format(split): total_loss.clone().detach(),
                   "{}/rec_loss".format(split): rec_loss.detach(),
                   "{}/l1_loss".format(split): l1_loss.detach(),
                   "{}/sam_loss".format(split): sam_loss.detach(),
                   "{}/gan_loss".format(split): -disc_fake.mean(0).view(1).detach(),
                   "{}/feature_loss".format(split): feature_loss.detach(),
                   }
            return total_loss, log
        
        if optimizer_idx == 1:
            # train discriminator
            # for params in self.discriminator.parameters():
            #     params.requires_grad = True
            disc_fake, fake_features = self.discriminator(torch.concat([reconstructions, cond], dim=1).detach())
            disc_real, real_features = self.discriminator(torch.concat([labels, cond], dim=1).detach())
            feature_loss = 0
            for k in range(len(fake_features)):
                feature_loss += F.mse_loss(real_features[k], fake_features[k])
            total_loss = disc_fake.mean(0).view(1) - disc_real.mean(0).view(1) - feature_loss * self.features_weight
            log = {"{}/total_loss".format(split): total_loss.clone().detach(),
                   "{}/disc_real".format(split): disc_real.mean(0).view(1).detach(),
                   "{}/disc_fake".format(split): disc_fake.mean(0).view(1).detach(),
                   "{}/feature_loss".format(split): feature_loss.detach(),
                   }
            return total_loss, log
        
class Wasserstein_Loss_features(nn.Module):
    def __init__(self, *, l1_weight, sam_weight, features_weight, disc_weight, perceptual_weight =0.01, threshold = 0, **kwargs) -> None:
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.l1_weight = l1_weight
        self.sam_loss = SamLoss()
        self.sam_weight = sam_weight
        self.features_weight = features_weight
        self.disc_weight = disc_weight
        self.threshold = threshold
        self.perceptual_weight = perceptual_weight
        self.perceptual_model = load_hsi_perceptual_encoder()

    def cal_perceptual_loss(self, reconstructions, labels):
        loss = 0
        real_features = self.perceptual_model.get_features(labels)
        fake_features = self.perceptual_model.get_features(reconstructions)
        for real_feature, fake_feature in zip(real_features, fake_features):
            loss += F.mse_loss(real_feature, fake_feature)
        return loss

    def calculate_adaptive_weight(self, rec_loss, g_loss, last_layer=None):
        if last_layer is not None:
            rec_grads = torch.autograd.grad(rec_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            rec_grads = torch.autograd.grad(rec_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(rec_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        # d_weight = d_weight * self.discriminator_weight
        return d_weight
    
    def forward(self, discriminator, reconstructions, labels, cond, optimizer_idx, global_step, last_layer = None):
        l1_loss = self.l1_loss(reconstructions, labels)
        sam_loss = self.sam_loss(reconstructions, labels)
        rec_loss = l1_loss * self.l1_weight + sam_loss * self.sam_weight
        if global_step > self.threshold:
            disc_factor = 1.0
        else:
            disc_factor = 0.0

        if optimizer_idx == 0:
            # train generator
            for params in discriminator.parameters():
                params.requires_grad = False
            disc_fake, fake_features = discriminator(torch.concat([reconstructions, cond], dim=1))
            disc_real, real_features = discriminator(torch.concat([labels, cond], dim=1))
            disc_fake = disc_fake.mean(0).view(1)
            disc_fake = disc_fake * self.calculate_adaptive_weight(rec_loss, disc_fake, last_layer) * disc_factor

            feature_loss = 0
            for k in range(len(fake_features)):
                feature_loss += F.mse_loss(real_features[k].detach(), fake_features[k])
            perceptual_loss = self.cal_perceptual_loss(reconstructions, labels) * self.perceptual_weight
            total_loss = - disc_fake + rec_loss + feature_loss * self.features_weight * disc_factor + perceptual_loss
            return total_loss
        
        if optimizer_idx == 1:
            # train discriminator
            for params in discriminator.parameters():
                params.requires_grad = True
            disc_fake, fake_features = discriminator(torch.concat([reconstructions, cond], dim=1).detach())
            disc_real, real_features = discriminator(torch.concat([labels, cond], dim=1).detach())
            feature_loss = 0
            for k in range(len(fake_features)):
                feature_loss += F.mse_loss(real_features[k], fake_features[k])
            total_loss = disc_fake.mean(0).view(1) - disc_real.mean(0).view(1) - feature_loss * self.features_weight
            return total_loss * disc_factor
        
class Wasserstein_Loss(nn.Module):
    def __init__(self, *, l1_weight, sam_weight, perceptual_weight =0.01, threshold = 0, **kwargs) -> None:
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.l1_weight = l1_weight
        self.sam_loss = SamLoss()
        self.sam_weight = sam_weight
        self.threshold = threshold
        self.perceptual_weight = perceptual_weight
        self.perceptual_model = load_hsi_perceptual_encoder()

    def cal_perceptual_loss(self, reconstructions, labels):
        loss = 0
        real_features = self.perceptual_model.get_features(labels)
        fake_features = self.perceptual_model.get_features(reconstructions)
        for real_feature, fake_feature in zip(real_features, fake_features):
            loss += F.mse_loss(real_feature, fake_feature)
        return loss

    def calculate_adaptive_weight(self, rec_loss, g_loss, last_layer=None):
        if last_layer is not None:
            rec_grads = torch.autograd.grad(rec_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            rec_grads = torch.autograd.grad(rec_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(rec_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        # d_weight = d_weight * self.discriminator_weight
        return d_weight
    
    def forward(self, discriminator, reconstructions, labels, cond, optimizer_idx, global_step, last_layer = None):
        l1_loss = self.l1_loss(reconstructions, labels)
        sam_loss = self.sam_loss(reconstructions, labels)
        rec_loss = l1_loss * self.l1_weight + sam_loss * self.sam_weight
        if global_step > self.threshold:
            disc_factor = 1.0
        else:
            disc_factor = 0.0


        if optimizer_idx == 0:
            # train generator
            for params in discriminator.parameters():
                params.requires_grad = False
            disc_fake = discriminator(torch.concat([reconstructions, cond], dim=1))
            disc_fake = disc_fake.mean()
            disc_fake = disc_fake * self.calculate_adaptive_weight(rec_loss, disc_fake, last_layer) * disc_factor
            perceptual_loss = self.cal_perceptual_loss(reconstructions, labels) * self.perceptual_weight
            total_loss = - disc_fake + rec_loss + perceptual_loss
            return total_loss
        
        if optimizer_idx == 1:
            # train discriminator
            for params in discriminator.parameters():
                params.requires_grad = True
            disc_fake = discriminator(torch.concat([reconstructions, cond], dim=1).detach())
            disc_real = discriminator(torch.concat([labels, cond], dim=1).detach())
            total_loss = disc_fake.mean() - disc_real.mean()
            # print(disc_fake.mean(), disc_real.mean())
            return total_loss * disc_factor
        
class LS_Loss(nn.Module):
    def __init__(self, *, l1_weight, sam_weight, perceptual_weight =0.01, threshold = 0, **kwargs) -> None:
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.l1_weight = l1_weight
        self.sam_loss = SamLoss()
        self.sam_weight = sam_weight
        self.criterion = nn.MSELoss()
        self.threshold = threshold
        self.perceptual_weight = perceptual_weight
        self.perceptual_model = load_hsi_perceptual_encoder()

    def cal_perceptual_loss(self, reconstructions, labels):
        loss = 0
        real_features = self.perceptual_model.get_features(labels)
        fake_features = self.perceptual_model.get_features(reconstructions)
        for real_feature, fake_feature in zip(real_features, fake_features):
            loss += F.mse_loss(real_feature, fake_feature)
        return loss

    def calculate_adaptive_weight(self, rec_loss, g_loss, last_layer=None):
        if last_layer is not None:
            rec_grads = torch.autograd.grad(rec_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            rec_grads = torch.autograd.grad(rec_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(rec_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        # d_weight = d_weight * self.discriminator_weight
        return d_weight
    
    def forward(self, discriminator, reconstructions, labels, cond, optimizer_idx, global_step, last_layer = None):
        l1_loss = self.l1_loss(reconstructions, labels)
        sam_loss = self.sam_loss(reconstructions, labels)
        rec_loss = l1_loss * self.l1_weight + sam_loss * self.sam_weight
        if global_step > self.threshold:
            disc_factor = 1.0
        else:
            disc_factor = 0.0

        if optimizer_idx == 0:
            # train generator
            for params in discriminator.parameters():
                params.requires_grad = False
            disc_fake = discriminator(torch.concat([reconstructions, cond], dim=1))
            real_labels = torch.ones_like(disc_fake).cuda()
            disc_fake = self.criterion(disc_fake, real_labels)
            disc_fake = disc_fake * self.calculate_adaptive_weight(rec_loss, disc_fake, last_layer) * disc_factor
            perceptual_loss = self.cal_perceptual_loss(reconstructions, labels) * self.perceptual_weight
            total_loss = - disc_fake + rec_loss + perceptual_loss
            return total_loss
        
        if optimizer_idx == 1:
            # train discriminator
            for params in discriminator.parameters():
                params.requires_grad = True
            disc_fake = discriminator(torch.concat([reconstructions, cond], dim=1).detach())
            disc_real = discriminator(torch.concat([labels, cond], dim=1).detach())
            real_labels = torch.ones_like(disc_fake).cuda()
            fake_labels = torch.zeros_like(disc_fake).cuda()
            disc_fake = self.criterion(disc_fake, fake_labels)
            disc_real = self.criterion(disc_real, real_labels)
            total_loss = disc_fake.mean(0).view(1) - disc_real.mean(0).view(1)
            return total_loss * disc_factor
