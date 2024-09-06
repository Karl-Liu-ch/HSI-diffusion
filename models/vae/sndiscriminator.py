import torch
import torch.nn as nn
import torch.nn.functional as F
from taming.modules.losses.vqperceptual import * 
from utils import *
from models.losses.gan_loss import GANLoss, Loss_MRAE, SAMLoss, LossDeltaE

class Huber_loss(nn.Module):
    def __init__(self):
        super(Huber_loss, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = F.l1_loss(outputs, label) + F.mse_loss(outputs, label)
        return error.mean()

class MRAE(nn.Module):
    def __init__(self):
        super(MRAE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        if label.all() == False:
            error = torch.abs(outputs - label) / (label + 1e-5)
        else:
            error = torch.abs(outputs - label) / label
        return error.mean()

class SAM_Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, preds, target):
        assert preds.shape == target.shape
        dot_product = (preds * target).sum(dim=1)
        preds_norm = preds.norm(dim=1)
        target_norm = target.norm(dim=1)
        sam_score = torch.clamp(dot_product / (preds_norm * target_norm), -1, 1).acos()
        return sam_score.mean()
        
def deltaE_Loss(fake, gt):
    loss = ciede2000_diff(rgb2lab_diff(gt, device), rgb2lab_diff(fake, device), device)
    color_loss=loss.mean()
    return color_loss

class Loss_DeltaE(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_hs2rgb = nn.Conv2d(31, 3, 1, bias=False)
        cie_matrix = CAM_FILTER
        cie_matrix = torch.from_numpy(np.transpose(cie_matrix, [1, 0])).unsqueeze(-1).unsqueeze(-1).float()
        self.model_hs2rgb.weight.data = cie_matrix
        self.model_hs2rgb.weight.requires_grad = False

    def forward(self, outputs, label, rgb_gt = None):
        # hs2rgb
        if rgb_gt is None:
            rgb_tensor = self.model_hs2rgb(outputs)
            rgb_tensor = normalization_image(rgb_tensor)
            rgb_label = self.model_hs2rgb(label)
            rgb_label = normalization_image(rgb_label)
        else:
            rgb_tensor = self.model_hs2rgb(outputs)
            rgb_tensor = normalization_image(rgb_tensor)
            rgb_label = normalization_image(rgb_gt)
        deltaE = deltaE_Loss(rgb_tensor, rgb_label)
        return deltaE

criterion_mrae = MRAE()

class SpectralNormalizationWDiscriminator(nn.Module):
    def __init__(self, disc_start, discconfig, l1_loss = 'mrae',l1_weight = 1.0, sam_weight = 1.0, deltaE_weight = 1.0, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0, 
                 disc_factor=1.0, disc_weight=1.0, features_weight=0.01,
                 perceptual_weight=1.0, disc_conditional=False, losstype = 'wasserstein', use_features = False, *args, **kwargs):
        super().__init__()
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        match l1_loss:
            case 'mrae':
                self.l1_loss = MRAE()
            case 'l1':
                self.l1_loss = nn.L1Loss()
            case 'huber':
                self.l1_loss = Huber_loss()
        self.l1_weight = l1_weight
        self.sam_loss = SAM_Loss()
        self.sam_weight = sam_weight
        self.deltaELoss = Loss_DeltaE()
        self.deltaE_weight = deltaE_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)
        self.criterionGAN = GANLoss(losstype)
        self.discriminator = instantiate_from_config(discconfig)
        self.discriminator_iter_start = disc_start
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.features_weight = features_weight
        self.use_features = use_features

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        # return 1.0
        return d_weight
    
    def disc_predict(self, reconstructions, labels):
        if self.use_features:
            disc_fake, disc_fake_features = self.discriminator(reconstructions)
            disc_real, disc_real_features = self.discriminator(labels)
        else:
            disc_fake = self.discriminator(reconstructions)
            disc_real = self.discriminator(labels)
            disc_real_features = None
            disc_fake_features = None
        return disc_real, disc_fake, disc_real_features, disc_fake_features

    def gan_loss(self, reconstructions, labels, cond = None):
        if cond is None:
            disc_real, disc_fake, disc_real_features, disc_fake_features = self.disc_predict(reconstructions, labels)
        else:
            disc_real, disc_fake, disc_real_features, disc_fake_features = self.disc_predict(torch.concat([reconstructions, cond], dim=1), torch.concat([labels, cond], dim=1))
        return disc_real, disc_fake, disc_real_features, disc_fake_features

    def recon_loss(self, reconstructions, labels):
        l1_loss = self.l1_loss(reconstructions, labels)
        rec_loss = l1_loss * self.l1_weight
        if self.sam_weight > 0:
            sam_loss = self.sam_loss(reconstructions, labels)
            rec_loss += sam_loss * self.sam_weight
        if self.deltaE_weight > 0:
            deltaELoss = self.deltaELoss(reconstructions, labels)
            rec_loss += deltaELoss * self.deltaE_weight
        return rec_loss

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                weights=None):
        rec_loss = self.recon_loss(reconstructions.contiguous(), inputs.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        # nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        # weighted_nll_loss = nll_loss
        # if weights is not None:
        #     weighted_nll_loss = weights*nll_loss
        # weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        # nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        # kl_loss = posteriors.kl()
        # kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        scale = reconstructions.shape[1] * reconstructions.shape[2] * reconstructions.shape[3]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0] / scale
        nll_loss = rec_loss
        weighted_nll_loss = nll_loss

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            disc_real, disc_fake, disc_real_features, disc_fake_features = self.gan_loss(reconstructions.contiguous(), inputs.contiguous(), cond)
            feature_loss = 0
            if self.use_features and self.features_weight > 0:
                for k in range(len(disc_fake_features)):
                    feature_loss += F.mse_loss(disc_real_features[k].detach(), disc_fake_features[k])
            g_loss = self.criterionGAN(disc_fake, target_is_real=True)

            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)

            loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * (g_loss + feature_loss * self.features_weight)

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(), "{}/logvar".format(split): self.logvar.detach(),
                   "{}/kl_loss".format(split): kl_loss.detach().mean(), "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight,
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            disc_real, disc_fake, disc_real_features, disc_fake_features = self.gan_loss(reconstructions.contiguous().detach(), inputs.contiguous().detach(), cond)
            feature_loss = 0
            if self.use_features and self.features_weight > 0:
                for k in range(len(disc_fake_features)):
                    feature_loss += F.mse_loss(disc_real_features[k], disc_fake_features[k])
            disc_real = self.criterionGAN(disc_real, target_is_real=True)
            disc_fake = self.criterionGAN(disc_fake, target_is_real=False)
            d_loss = disc_fake + disc_real - feature_loss * self.features_weight

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss *= disc_factor

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): disc_real.detach().mean(),
                   "{}/logits_fake".format(split): disc_fake.detach().mean()
                   }
            return d_loss, log