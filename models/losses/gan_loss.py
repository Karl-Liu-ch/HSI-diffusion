import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from models.gan.networks import *
from utils import instantiate_from_config

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
        
class GanLoss_features(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.criterion = nn.L1Loss()
        self.sam_loss = SamLoss()

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
    
    def forward(self, idx, discriminator, realAB, fakeAB, reconstruction, label, l1_weight, sam_weight, feature_weight, disc_weight, last_layer):
        if idx == 0:
            D_real, D_real_feature = discriminator(realAB)
            loss_real = -D_real.mean(0).view(1)
            D_fake, D_fake_feature = self.D(fakeAB.detach())
            loss_fake = D_fake.mean(0).view(1)
            perceptual_loss = 0
            for k in range(len(D_fake_feature)):
                perceptual_loss -= F.mse_loss(D_real_feature[k], D_fake_feature[k])
            loss_d = loss_real + loss_fake + perceptual_loss * disc_weight
            return loss_d
        
        elif idx == 1:
            pred_fake, D_fake_feature = discriminator(fakeAB)
            loss_G = -pred_fake.mean(0).view(1)
            lossl1 = self.criterion(reconstruction, label) * l1_weight
            losssam = self.sam_loss(reconstruction, label) * sam_weight
            perceptual_loss = 0
            for k in range(len(D_fake_feature)):
                perceptual_loss += F.mse_loss(D_real_feature[k].detach(), D_fake_feature[k])
            weight_gan = self.calculate_adaptive_weight(lossl1 + losssam, loss_G, last_layer)
            loss_G *= weight_gan
            loss_G += lossl1 + losssam + perceptual_loss * feature_weight * weight_gan / len(D_fake_feature)
            return loss_G
        
class GanLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.criterion = nn.L1Loss()
        self.sam_loss = SamLoss()

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
    
    def forward(self, idx, discriminator, realAB, fakeAB, reconstruction, label, l1_weight, sam_weight, last_layer):
        if idx == 0:
            D_real = discriminator(realAB)
            loss_real = -D_real.mean(0).view(1)
            D_fake = self.D(fakeAB.detach())
            loss_fake = D_fake.mean(0).view(1)
            loss_d = loss_real + loss_fake
            return loss_d
        elif idx == 1:
            pred_fake = discriminator(fakeAB)
            loss_G = -pred_fake.mean(0).view(1)
            lossl1 = self.criterion(reconstruction, label) * l1_weight
            losssam = self.sam_loss(reconstruction, label) * sam_weight
            weight_gan = self.calculate_adaptive_weight(lossl1 + losssam, loss_G, last_layer)
            loss_G *= weight_gan
            loss_G += lossl1 + losssam
            return loss_G