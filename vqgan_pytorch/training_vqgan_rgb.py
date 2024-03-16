import sys
sys.path.append('./')
import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import utils as vutils
from vqgan_pytorch.discriminator import Discriminator
from vqgan_pytorch.lpips import LPIPS
from vqgan_pytorch.vqgan import VQGAN
# from vqgan_pytorch.utils import weights_init
from dataset.datasets import *
from utils import *
from differential_color_functions import rgb2lab_diff, ciede2000_diff
from einops import repeat
from omegaconf import OmegaConf
config_path = "configs/hsi_vqgan.yaml"
cfg = OmegaConf.load(config_path)
ddconfig = cfg.model.cond_stage.params.ddconfig

criterion_mrae = Loss_MRAE()
criterion_rmse = Loss_RMSE()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalization_image(x):
    B, C, H, W = x.shape
    v_min = x.view(B, -1).min(dim=1)[0]
    v_max = x.view(B, -1).max(dim=1)[0]
    v_min = repeat(v_min, 'b -> b c h w', c = C, h = H, w = W)
    v_max = repeat(v_max, 'b -> b c h w', c = C, h = H, w = W)
    x_norm = (x - v_min) / (v_max - v_min)
    return x_norm

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class TrainVQGAN:
    def __init__(self, args):
        self.vqgan = VQGAN(args, ddconfig).to(device=args.device)
        self.discriminator = Discriminator(args).to(device=args.device)
        self.discriminator.apply(weights_init)
        self.perceptual_loss = LPIPS().eval().to(device=args.device)
        self.opt_vq, self.opt_disc, self.schedulervq, self.schedulerdisc = self.configure_optimizers(args)
        # ckpt = torch.load(os.path.join("/work3/s212645/Spectral_Reconstruction/checkpoint/VQGAN/", f"vqgan_rgb_epoch_latest.pt"))
        # self.vqgan.load_state_dict(ckpt)

        self.prepare_training()

        self.train(args)

    def configure_optimizers(self, args):
        lr = args.learning_rate
        opt_vq = torch.optim.Adam(
            list(self.vqgan.encoder.parameters()) +
            list(self.vqgan.decoder.parameters()) +
            list(self.vqgan.codebook.parameters()) +
            list(self.vqgan.quant_conv.parameters()) +
            list(self.vqgan.post_quant_conv.parameters()),
            lr=lr, eps=1e-08, betas=(args.beta1, args.beta2)
        )
        schedulervq = torch.optim.lr_scheduler.CosineAnnealingLR(opt_vq, args.epochs, eta_min=1e-6)
        opt_disc = torch.optim.Adam(self.discriminator.parameters(),
                                    lr=lr, eps=1e-08, betas=(args.beta1, args.beta2))
        schedulerdisc = torch.optim.lr_scheduler.CosineAnnealingLR(opt_disc, args.epochs, eta_min=1e-6)

        return opt_vq, opt_disc, schedulervq, schedulerdisc

    @staticmethod
    def prepare_training():
        os.makedirs("results", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)

    def train(self, args):
        # train_dataset = load_data(args)
        train_data = TrainDataset(data_root=args.data_root, crop_size=args.image_size, valid_ratio = 0.1, test_ratio=0.1)
        train_dataset = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=2,
                                pin_memory=True, drop_last=True)
        steps_per_epoch = len(train_dataset)
        for epoch in range(args.epochs):
            with tqdm(range(len(train_dataset))) as pbar:
                for i, (_, _, imgs) in zip(pbar, train_dataset):
                    imgs = imgs.to(device=args.device)
                    for p in self.vqgan.parameters():
                        p.requires_grad = True
                    decoded_images, _, q_loss = self.vqgan(imgs)

                    disc_real = self.discriminator(imgs)
                    disc_fake = self.discriminator(decoded_images)

                    disc_factor = self.vqgan.adopt_weight(args.disc_factor, epoch*steps_per_epoch+i, threshold=args.disc_start)

                    perceptual_loss = self.perceptual_loss(imgs, decoded_images).mean()
                    rec_loss = F.l1_loss(decoded_images, imgs) + F.mse_loss(decoded_images, imgs)
                    perceptual_rec_loss = args.perceptual_loss_factor * perceptual_loss + args.rec_loss_factor * rec_loss
                    # perceptual_rec_loss = perceptual_rec_loss.mean()
                    g_loss = -torch.mean(disc_fake)

                    λ = self.vqgan.calculate_lambda(perceptual_rec_loss, g_loss)
                    vq_loss = perceptual_rec_loss + q_loss + disc_factor * λ * g_loss

                    d_loss_real = torch.mean(F.relu(1. - disc_real))
                    d_loss_fake = torch.mean(F.relu(1. + disc_fake))
                    gan_loss = disc_factor * 0.5*(d_loss_real + d_loss_fake)

                    self.opt_vq.zero_grad()
                    vq_loss.backward(retain_graph=True)

                    self.opt_disc.zero_grad()
                    gan_loss.backward()

                    self.opt_vq.step()
                    self.opt_disc.step()
                    
                    pbar.set_postfix(
                        VQ_Loss=np.round(vq_loss.cpu().detach().numpy().item(), 5),
                        GAN_Loss=np.round(gan_loss.cpu().detach().numpy().item(), 3)
                    )
                    pbar.update(0)
                print(rec_loss.item())
                torch.save(self.vqgan.state_dict(), os.path.join("/work3/s212645/Spectral_Reconstruction/checkpoint/VQGAN/", f"vqgan_rgb_epoch_latest.pt"))
                self.schedulervq.step()
                self.schedulerdisc.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--image-size', type=int, default=256, help='Image height and width (default: 256)')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors (default: 256)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images (default: 3)')
    parser.add_argument('--dataset-path', type=str, default='/data', help='Path to data (default: /data)')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=16, help='Input batch size for training (default: 6)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=5e-06, help='Learning rate (default: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param (default: 0.0)')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param (default: 0.999)')
    parser.add_argument('--disc-start', type=int, default=0, help='When to start the discriminator (default: 0)')
    parser.add_argument('--disc-factor', type=float, default=1., help='')
    parser.add_argument('--rec-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1., help='Weighting factor for perceptual loss.')
    parser.add_argument("--data_root", type=str, default='/work3/s212645/Spectral_Reconstruction/')

    args = parser.parse_args()
    # args.dataset_path = r"C:\Users\dome\datasets\flowers"

    train_vqgan = TrainVQGAN(args)



