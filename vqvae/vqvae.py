import sys
sys.path.append('./')
from options import opt
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset.datasets import *
import einops

class ResidualBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        tmp = self.relu(x)
        tmp = self.conv1(tmp)
        tmp = self.relu(tmp)
        tmp = self.conv2(tmp)
        return x + tmp
    
class VQVAE(nn.Module):

    def __init__(self, input_dim, dim, n_embedding):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(input_dim, dim, 4, 2, 1),
                                     nn.ReLU(), nn.Conv2d(dim, dim, 4, 2, 1),
                                     nn.ReLU(), nn.Conv2d(dim, dim, 3, 1, 1),
                                     ResidualBlock(dim), ResidualBlock(dim))
        self.vq_embedding = nn.Embedding(n_embedding, dim)
        self.vq_embedding.weight.data.uniform_(-1.0 / n_embedding,
                                               1.0 / n_embedding)
        self.decoder = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            ResidualBlock(dim), ResidualBlock(dim),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(dim, input_dim, 4, 2, 1))
        self.n_downsample = 2
    
    def forward(self, x):
        # encode
        ze = self.encoder(x)
        
        # ze: [N, C, H, W]
        # embedding [K, C]
        embedding = self.vq_embedding.weight.data
        N, C, H, W = ze.shape
        K, _ = embedding.shape
        embedding_broadcast = embedding.reshape(1, K, C, 1, 1)
        ze_broadcast = ze.reshape(N, 1, C, H, W)
        distance = torch.sum((embedding_broadcast - ze_broadcast)**2, 2)
        nearest_neighbor = torch.argmin(distance, 1)
        # make C to the second dim
        zq = self.vq_embedding(nearest_neighbor).permute(0, 3, 1, 2)
        # stop gradient
        decoder_input = ze + (zq - ze).detach()
        
        # decode
        x_hat = self.decoder(decoder_input)
        return x_hat, ze, zq

datanames = ['ARAD/']
train_data = TrainDataset(data_root=opt.data_root, crop_size=opt.patch_size, valid_ratio = 0.1, test_ratio=0.1, datanames = datanames)
train_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True, num_workers=2,
                        pin_memory=True, drop_last=True)
val_data = ValidDataset(data_root=opt.data_root, crop_size=opt.patch_size, valid_ratio = 0.1, test_ratio=0.1, datanames = datanames)
val_loader = DataLoader(dataset=val_data, batch_size=opt.batch_size, shuffle=True, num_workers=2, pin_memory=True)

def train_vqvae(model: VQVAE,
                dataloader=train_loader,
                img_shape=None,
                device='cuda',
                batch_size=64,
                lr=1e-3,
                n_epochs=100,
                l_w_embedding=1,
                l_w_commitment=0.25):
    print('batch size:', batch_size)
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    for e in range(n_epochs):
        total_loss = 0

        for step, (_, x, _) in enumerate(dataloader):
            current_batch_size = x.shape[0]
            x = x.to(device)

            x_hat, ze, zq = model(x)
            l_reconstruct = l1_loss(x, x_hat)
            l_embedding = mse_loss(ze.detach(), zq)
            l_commitment = mse_loss(ze, zq.detach())
            loss = l_reconstruct + \
                l_w_embedding * l_embedding + l_w_commitment * l_commitment
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * current_batch_size
        total_loss /= len(dataloader.dataset)
        print(f'epoch {e} loss: {total_loss}')
    print('Done')

def reconstruct(model, val, device='cuda'):
    model.to(device)
    model.eval()
    errors = []
    with torch.no_grad():
        for step, (_, x, _) in enumerate(val):
            x_hat, _, _ = model(x)
            errors.append(F.l1_loss(x, x_hat).cpu().numpy())
    errors = np.array(errors)
    return errors.mean()

vqvae = VQVAE(31, 64, 64)
train_vqvae(vqvae)
error = reconstruct(vqvae, val_loader)
print(error)