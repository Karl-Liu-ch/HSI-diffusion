import sys
sys.path.append('./')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from models.gan.networks import *
from utils import *
from models.vae.networks import Encoder
from omegaconf import OmegaConf
from torchmetrics.image import StructuralSimilarityIndexMeasure, MultiScaleStructuralSimilarityIndexMeasure
from taming.modules.losses.vqperceptual import * 
from utils import instantiate_from_config

class SSIMLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        
    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        ssim_score = self.ssim(outputs, label)
        ssim_score = torch.mean(ssim_score.view(-1))
        error = 1.0 - ssim_score
        return error
    
    def reset(self):
        self.ssim.reset()

def load_hsi_perceptual_encoder():
    ckpt_path = '/work3/s212645/Spectral_Reconstruction/checkpoint/vae_perceptual/lightning_logs/version_0/checkpoints/epoch125-mrae_avg0.05.ckpt'
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
        # return 1.0
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
            disc_fake = - disc_fake.mean(0).view(1)
            disc_fake *= self.calculate_adaptive_weight(rec_loss, disc_fake, last_layer)

            feature_loss = 0
            for k in range(len(fake_features)):
                feature_loss += F.mse_loss(real_features[k].detach(), fake_features[k])
            total_loss = disc_fake + rec_loss + feature_loss * self.features_weight
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


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label).to(device))
        self.register_buffer('fake_label', torch.tensor(target_fake_label).to(device))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'wasserstein':
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wasserstein':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

class Loss(nn.Module):
    def __init__(self, *, l1_weight, sam_weight, features_weight, disc_weight, perceptual_weight =0.0, deltaE_weight = 1.0, threshold = 0, losstype = 'wasserstein', **kwargs) -> None:
        super().__init__()
        # self.l1_loss = nn.L1Loss()
        self.l1_loss = Loss_MRAE()
        self.l1_weight = l1_weight
        self.sam_loss = SamLoss()
        self.sam_weight = sam_weight
        self.deltaELoss = LossDeltaE().cuda()
        self.deltaE_weight = deltaE_weight
        self.features_weight = features_weight
        self.disc_weight = disc_weight
        self.threshold = threshold
        self.perceptual_weight = perceptual_weight
        self.criterionGAN = GANLoss(losstype)
        if perceptual_weight > 0:
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
            return 1.0

        d_weight = torch.norm(rec_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return 1.0
        return d_weight
    
    def recon_loss(self, reconstructions, labels):
        l1_loss = self.l1_loss(reconstructions, labels)
        sam_loss = self.sam_loss(reconstructions, labels)
        deltaELoss = self.deltaELoss(reconstructions, labels)
        rec_loss = l1_loss * self.l1_weight + sam_loss * self.sam_weight + deltaELoss * self.deltaE_weight
        return rec_loss

    def gan_loss(self, discriminator, reconstructions, labels, cond):
        disc_fake = discriminator(torch.concat([reconstructions, cond], dim=1))
        disc_real = discriminator(torch.concat([labels, cond], dim=1))
        disc_real_features = None
        disc_fake_features = None
        return disc_real, disc_fake, disc_real_features, disc_fake_features

    def adopt_factor(self, global_step):
        if global_step >= self.threshold:
            disc_factor = 1.0
        else:
            disc_factor = 0.0
        return disc_factor
    
    def train_gen(self, discriminator, reconstructions, labels, cond, global_step, last_layer = None):
        rec_loss = self.recon_loss(reconstructions, labels)
        disc_factor = self.adopt_factor(global_step)
        for params in discriminator.parameters():
            params.requires_grad = False
        disc_real, disc_fake, real_features, fake_features = self.gan_loss(discriminator, reconstructions, labels, cond)
        disc_fake = self.criterionGAN(disc_fake, target_is_real=True)
        # disc_fake = - disc_fake
        disc_fake = disc_fake * self.calculate_adaptive_weight(rec_loss, disc_fake, last_layer) * disc_factor
        feature_loss = 0
        if (real_features is not None) and (fake_features is not None):
            for k in range(len(fake_features)):
                feature_loss += F.mse_loss(real_features[k].detach(), fake_features[k])
        if self.perceptual_weight > 0:
            perceptual_loss = self.cal_perceptual_loss(reconstructions, labels) * self.perceptual_weight
        else:
            perceptual_loss = 0
        total_loss = disc_fake + rec_loss + feature_loss * self.features_weight * disc_factor + perceptual_loss
        log = {'gen loss':disc_fake, 'recon loss':rec_loss, 'delta e': self.deltaELoss(reconstructions, labels)}
        return total_loss, log
    
    def train_disc(self, discriminator, reconstructions, labels, cond, global_step):
        disc_factor = self.adopt_factor(global_step)
        for params in discriminator.parameters():
            params.requires_grad = True
        disc_real, disc_fake, real_features, fake_features = self.gan_loss(discriminator, reconstructions.detach(), labels.detach(), cond.detach())
        disc_real = self.criterionGAN(disc_real, target_is_real=True)
        disc_fake = self.criterionGAN(disc_fake, target_is_real=False)
        feature_loss = 0
        if (real_features is not None) and (fake_features is not None):
            for k in range(len(fake_features)):
                feature_loss += F.mse_loss(real_features[k], fake_features[k])
        total_loss = disc_fake + disc_real - feature_loss * self.features_weight
        log = {'real loss':disc_real, 'fake loss':disc_fake}
        return total_loss * disc_factor, log

    def forward(self, discriminator, reconstructions, labels, cond, global_step, mode, last_layer = None):
        if mode == 'gen':
            total_loss = self.train_gen(discriminator, reconstructions, labels, cond, global_step, last_layer)
            return total_loss
        
        elif mode == 'dics':
            total_loss = self.train_disc(discriminator, reconstructions, labels, cond, global_step)
            return total_loss
        
class Wasserstein_Loss_features(Loss):
    def gan_loss(self, discriminator, reconstructions, labels, cond):
        disc_fake, disc_fake_features = discriminator(torch.concat([reconstructions, cond], dim=1))
        disc_real, disc_real_features = discriminator(torch.concat([labels, cond], dim=1))
        return disc_real.mean(), disc_fake.mean(), disc_real_features, disc_fake_features
    

class SpectralNormalizationWDiscriminator(nn.Module):
    def __init__(self, disc_start, discconfig, l1_weight = 1.0, sam_weight = 1.0, deltaE_weight = 1.0, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0, 
                 disc_factor=1.0, disc_weight=1.0, features_weight=0.01,
                 perceptual_weight=1.0, disc_conditional=False, losstype = 'wasserstein', use_features = False, *args, **kwargs):
        super().__init__()
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        self.l1_loss = Loss_MRAE()
        self.l1_weight = l1_weight
        self.sam_loss = SAMLoss()
        self.sam_weight = sam_weight
        self.deltaELoss = LossDeltaE()
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
        # rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        rec_loss = self.recon_loss(reconstructions.contiguous(), inputs.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

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
                   "{}/d_weight".format(split): d_weight.detach(),
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