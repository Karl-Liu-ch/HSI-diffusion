import sys
sys.path.append('./')
import os
import argparse
import torch
from torchvision import utils as vutils
from transformer import VQGANTransformer
from tqdm import tqdm
from dataset.datasets import *
# from options import opt
from utils import *
criterion_mrae = Loss_MRAE()

parser = argparse.ArgumentParser(description="VQGAN")
parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z.')
parser.add_argument('--image-size', type=int, default=256, help='Image height and width.)')
parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors.')
parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar.')
parser.add_argument('--image-channels', type=int, default=31, help='Number of channels of images.')
parser.add_argument('--dataset-path', type=str, default='./data', help='Path to data.')
parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/last_ckpt.pt',
                    help='Path to checkpoint.')
parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
parser.add_argument('--batch-size', type=int, default=20, help='Input batch size for training.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate.')
parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param.')
parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param.')
parser.add_argument('--disc-start', type=int, default=10000, help='When to start the discriminator.')
parser.add_argument('--disc-factor', type=float, default=1., help='Weighting factor for the Discriminator.')
parser.add_argument('--l2-loss-factor', type=float, default=1.,
                    help='Weighting factor for reconstruction loss.')
parser.add_argument('--perceptual-loss-factor', type=float, default=1.,
                    help='Weighting factor for perceptual loss.')

parser.add_argument('--pkeep', type=float, default=0.5, help='Percentage for how much latent codes to keep.')
parser.add_argument('--sos-token', type=int, default=0, help='Start of Sentence token.')
parser.add_argument("--data_root", type=str, default='/work3/s212645/Spectral_Reconstruction/')

args = parser.parse_args()
args.checkpoint_path = "/work3/s212645/Spectral_Reconstruction/checkpoint/VQGAN/vqgan_epoch_latest.pt"

n = 100
epoch = 3
transformer = VQGANTransformer(args).to("cuda")
transformer.load_state_dict(torch.load(os.path.join("/work3/s212645/Spectral_Reconstruction/checkpoint/VQGAN/", f"transformer_latest.pt")))
print("Loaded state dict of Transformer")

test_data = TestDataset(data_root=args.data_root, crop_size=args.image_size, valid_ratio = 0.1, test_ratio=0.01)
print("Test set samples: ", len(test_data))
test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

for i, (rgb, label, _) in enumerate(test_loader):
    rgb = rgb.to(device=args.device)
    label = label.to(device=args.device)
    quant_z_c, indices_c, _ = transformer.vqgan.encode(transformer.convin(rgb))
    cond = indices_c.view(quant_z_c.shape[0], -1)
    start_indices = torch.zeros((args.batch_size, 0)).long().to("cuda")
    sos_tokens = torch.ones(start_indices.shape[0], 1) * 0
    sos_tokens = sos_tokens.long().to("cuda")
    sample_indices = transformer.sample(start_indices, sos_tokens, cond, steps=256)
    sampled_imgs = transformer.z_to_image(sample_indices, 16, 16)
    print(torch.abs(sampled_imgs - label).mean())
    print(criterion_mrae(sampled_imgs, label).item())