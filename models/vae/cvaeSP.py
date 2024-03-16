import sys
sys.path.append('./')
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.optim as optim
import torch.backends.cudnn as cudnn
from dataset.datasets import TrainDataset, ValidDataset
from utils import AverageMeter, record_loss, Loss_MRAE, Loss_RMSE, Loss_PSNR, Loss_Fid, Loss_SAM, Loss_SSIM, reconRGB, Loss_SID
from options import opt
import os
from torch.utils.data import DataLoader
from torch.autograd import Variable
from Models.VAE.Base import CVAECPModel
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
if opt.multigpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConditionalVAESP(nn.Module):

    def __init__(self,
                 in_channels: int,
                 condition_channels: int,
                 latent_dim: int,
                 hidden_dims,
                 img_size:int = 128,
                 **kwargs) -> None:
        super(ConditionalVAESP, self).__init__()

        self.latent_dim = latent_dim
        self.img_size = img_size
        self.in_channels = in_channels

        def ConvT(input_nums, output_nums):
            layer = []
            layer.append(nn.ConvTranspose2d(input_nums, output_nums, kernel_size=(4,4), stride=(2,2), padding=(1,1)))
            layer.append(nn.BatchNorm2d(output_nums))
            layer.append(nn.ReLU())
            return layer
        
        def Conv(input_nums, output_nums, stride):
            layer = []
            layer.append(nn.Conv2d(input_nums, output_nums, kernel_size=(3,3), stride=stride, padding=(1,1)))
            layer.append(nn.BatchNorm2d(output_nums))
            layer.append(nn.ReLU())
            return layer
        
        if hidden_dims is None:
            hidden_dims = [64, 128]
            
        self.hidden_dims = hidden_dims
        
        self.embed_class0 = nn.Conv2d(condition_channels, condition_channels, kernel_size=1)
        self.embed_class1 = nn.Sequential(*Conv(condition_channels, hidden_dims[0], stride=2))
        self.embed_class2 = nn.Sequential(*Conv(hidden_dims[0], hidden_dims[1], stride=2))
        
        self.embed_data = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # in_channels += condition_channels # To account for the extra label channel
        # Build Encoder
        self.encconv1 = nn.Sequential(*Conv(in_channels, hidden_dims[0], stride=2))
        self.encconv2 = nn.Sequential(*Conv(hidden_dims[0], hidden_dims[1], stride=2))
        
        self.hidden_size = (self.img_size // (2 ** len(hidden_dims)))
        
        
        self.fc_mu0 = nn.Sequential(*Conv(in_channels, in_channels, stride=1), 
                                    nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
                                    )
        self.fc_var0 = nn.Sequential(*Conv(in_channels, in_channels, stride=1), 
                                    nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
                                    )
        self.fc_mu1 = nn.Sequential(*Conv(hidden_dims[0], hidden_dims[0], stride=1), 
                                    nn.Conv2d(hidden_dims[0], hidden_dims[0], kernel_size=1, bias=False)
                                    )
        self.fc_var1 = nn.Sequential(*Conv(hidden_dims[0], hidden_dims[0], stride=1), 
                                    nn.Conv2d(hidden_dims[0], hidden_dims[0], kernel_size=1, bias=False)
                                    )
        self.fc_mu2 = nn.Sequential(*Conv(hidden_dims[1], hidden_dims[1], stride=1), 
                                    nn.Conv2d(hidden_dims[1], hidden_dims[1], kernel_size=1, bias=False)
                                    )
        self.fc_var2 = nn.Sequential(*Conv(hidden_dims[1], hidden_dims[1], stride=1), 
                                    nn.Conv2d(hidden_dims[1], hidden_dims[1], kernel_size=1, bias=False)
                                    )

        # Build Decoder

        self.decoder_input0 = nn.Sequential(*Conv(hidden_dims[1] * 2, hidden_dims[1] * 2, stride=1))
        self.decoder_input1 = nn.Sequential(*Conv(hidden_dims[0] * 2, hidden_dims[0] * 2, stride=1))
        self.decoder_input2 = nn.Sequential(*Conv(in_channels + condition_channels, in_channels + condition_channels, stride=1))

        self.decconv1 = nn.Sequential(*ConvT(hidden_dims[1] * 2, hidden_dims[1]))
        self.decconv2 = nn.Sequential(*ConvT(hidden_dims[1] * 2, in_channels + condition_channels))

        self.final_layer = nn.Sequential(
                            nn.Conv2d((in_channels + condition_channels) * 2, out_channels= self.in_channels,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        embed_input = self.embed_data(input)
        
        embed_input1 = self.encconv1(embed_input)
        embed_input2 = self.encconv2(embed_input1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu0 = self.fc_mu0(embed_input)
        log_var0 = self.fc_var0(embed_input) # [31, H, W]
        mu1 = self.fc_mu1(embed_input1)
        log_var1 = self.fc_var1(embed_input1) # [64, H / 2, W / 2]
        mu2 = self.fc_mu2(embed_input2)
        log_var2 = self.fc_var2(embed_input2) # [128, H / 4, W / 4]

        return [mu0, log_var0, mu1, log_var1, mu2, log_var2]

    def decode(self, z0, z1, z2):
        # z0: [256, H / 4, W / 4] torch.concat([z0, y2])
        result0 = self.decoder_input0(z0) # result0: [256, H / 4, H / 4]
        result0 = self.decconv1(result0) # result0: [128, H / 2, W / 2]
        # z1: [128, H / 2, W / 2] torch.concat([z1, y1])
        result1 = self.decoder_input1(z1) # result1 : [128, H / 2, W / 2]
        result1 = torch.concat([result1, result0], dim = 1) # result1: [256, H / 2, W / 2]
        result1 = self.decconv2(result1) # result1: [34, H, W]
        # z2: [34, H, W]
        result2 = self.decoder_input2(z2) # result2: [34, H, W]
        result2 = torch.concat([result2, result1], dim = 1) # result2: [68, H, W]
        result = self.final_layer(result2)
        return result

    def reparameterize(self, mu0, logvar0, mu1, logvar1, mu2, logvar2):
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std0 = torch.exp(0.5 * logvar0)
        eps0 = torch.randn_like(std0)
        std1 = torch.exp(0.5 * logvar1)
        eps1 = torch.randn_like(std1)
        std2 = torch.exp(0.5 * logvar2)
        eps2 = torch.randn_like(std2)
        return [eps0 * std0 + mu0, eps1 * std1 + mu1, eps2 * std2 + mu2]

    def forward(self, input, y):
        
        embed_y0 = self.embed_class0(y)
        embed_y1 = self.embed_class1(embed_y0)
        embed_y2 = self.embed_class2(embed_y1)
        embedded_input = self.embed_data(input)

        [mu0, log_var0, mu1, log_var1, mu2, log_var2] = self.encode(embedded_input)

        [z2, z1, z0] = self.reparameterize(mu0, log_var0, mu1, log_var1, mu2, log_var2)
        
        z0 = torch.cat([z0, embed_y2], dim = 1)
        z1 = torch.cat([z1, embed_y1], dim = 1)
        z2 = torch.cat([z2, embed_y0], dim = 1)
        
        recon = self.decode(z0, z1, z2)
        
        return  [recon, input, mu0, log_var0, mu1, log_var1, mu2, log_var2]

    def loss_function(self, recons, input, mu0, log_var0, mu1, log_var1, mu2, log_var2):

        # kld_weight = 3e-3  # Account for the minibatch samples from the dataset
        # kld_weight = 1024 / 128 / 128 / 31  # Account for the minibatch samples from the dataset
        kld_weight = 1 / 128 / 128  # Account for the minibatch samples from the dataset
        recons_loss =F.l1_loss(recons, input)
        
        Mu0 = rearrange(mu0, 'b c h w -> b (c h w)')
        Log_var0 = rearrange(log_var0, 'b c h w -> b (c h w)')
        kld_weight0 = 1 / Mu0.shape[1]
        Mu1 = rearrange(mu1, 'b c h w -> b (c h w)')
        Log_var1 = rearrange(log_var1, 'b c h w -> b (c h w)')
        kld_weight1 = 1 / Mu1.shape[1]
        Mu2 = rearrange(mu2, 'b c h w -> b (c h w)')
        Log_var2 = rearrange(log_var2, 'b c h w -> b (c h w)')
        kld_weight2 = 1 / Mu2.shape[1]

        kld_loss0 = torch.mean(-0.5 * torch.sum(1 + Log_var0 - Mu0 ** 2 - Log_var0.exp(), dim = 1), dim = 0)
        kld_loss1 = torch.mean(-0.5 * torch.sum(1 + Log_var1 - Mu1 ** 2 - Log_var1.exp(), dim = 1), dim = 0)
        kld_loss2 = torch.mean(-0.5 * torch.sum(1 + Log_var2 - Mu2 ** 2 - Log_var2.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * (kld_loss0 + kld_loss1 + kld_loss2)
        kld_loss = kld_loss0 + kld_loss1 + kld_loss2
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss * kld_weight}

    def sample(self, y):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        embed_y0 = self.embed_class0(y)
        embed_y1 = self.embed_class1(embed_y0)
        embed_y2 = self.embed_class2(embed_y1)
        z2 = torch.randn(y.shape[0], 31, y.shape[2], y.shape[3]).cuda()
        z1 = torch.randn_like(embed_y1).cuda()
        z0 = torch.randn_like(embed_y2).cuda()
        
        z0 = torch.cat([z0, embed_y2], dim = 1)
        z1 = torch.cat([z1, embed_y1], dim = 1)
        z2 = torch.cat([z2, embed_y0], dim = 1)
        
        samples = self.decode(z0, z1, z2)
        return samples

if __name__ == '__main__':
    model = ConditionalVAESP(in_channels=31, condition_channels=3, latent_dim=1024, hidden_dims=[64, 128], img_size=128)
    model_name = 'CVAESP'   
    spec = CVAECPModel(opt, model, model_name, multiGPU=opt.multigpu)
    if opt.loadmodel:
        try:
            spec.load_checkpoint()
        except:
            print('pretrained model loading failed')
    if opt.mode == 'train':
        spec.train()
        spec.test()
    elif opt.mode == 'test':
        spec.test()