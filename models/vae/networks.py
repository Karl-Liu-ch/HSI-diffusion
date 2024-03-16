import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.nn import functional as F
from Models.GAN.networks import ResnetBlock

class Encoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 condition_channels: int, 
                 n_blocks = 0) -> None:
        super(Encoder, self).__init__()

        self.in_channels = in_channels

        def ConvT(input_nums, output_nums):
            layer = []
            layer.append(nn.ConvTranspose2d(input_nums, output_nums, kernel_size=(4,4), stride=(2,2), padding=(1,1), bias=False))
            layer.append(nn.BatchNorm2d(output_nums))
            layer.append(nn.ReLU())
            return layer
        
        def Conv(input_nums, output_nums, stride):
            layer = []
            layer.append(nn.Conv2d(input_nums, output_nums, kernel_size=(3,3), stride=stride, padding=(1,1), bias=False))
            layer.append(nn.BatchNorm2d(output_nums))
            layer.append(nn.ReLU())
            return layer
        
        hidden_dims = [64, 128]
        self.embed_class0 = nn.Conv2d(condition_channels, condition_channels, kernel_size=1, bias=False)
        self.embed_class1 = nn.Sequential(*Conv(condition_channels, hidden_dims[0], stride=2))
        self.embed_class2 = nn.Sequential(*Conv(hidden_dims[0], hidden_dims[1], stride=2))
        
        self.embed_data = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
            
        self.hidden_dims = hidden_dims
        # Build Encoder
        self.encconv1 = nn.Sequential(*Conv(in_channels, hidden_dims[0], stride=2))
        self.encconv2 = nn.Sequential(*Conv(hidden_dims[0], hidden_dims[1], stride=2))
        
        # fc_mu0 = Conv(in_channels, in_channels, stride=1)
        # for i in range(n_blocks):
        #     fc_mu0 += [ResnetBlock(in_channels, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False)]
        # fc_mu0 += [nn.Conv2d(in_channels, in_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)]
        # self.fc_mu0 = nn.Sequential(*fc_mu0)
        # fc_var0 = Conv(in_channels, in_channels, stride=1)
        # for i in range(n_blocks):
        #     fc_var0 += [ResnetBlock(in_channels, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False)]
        # fc_var0 += [nn.Conv2d(in_channels, in_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)]
        # self.fc_var0 = nn.Sequential(*fc_var0)
        
        # fc_mu1 = Conv(hidden_dims[0], hidden_dims[0], stride=1)
        # for i in range(n_blocks):
        #     fc_mu1 += [ResnetBlock(hidden_dims[0], padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False)]
        # fc_mu1 += [nn.Conv2d(hidden_dims[0], hidden_dims[0], kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)]
        # self.fc_mu1 = nn.Sequential(*fc_mu1)
        # fc_var1 = Conv(hidden_dims[0], hidden_dims[0], stride=1)
        # for i in range(n_blocks):
        #     fc_var1 += [ResnetBlock(hidden_dims[0], padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False)]
        # fc_var1 += [nn.Conv2d(hidden_dims[0], hidden_dims[0], kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)]
        # self.fc_var1 = nn.Sequential(*fc_var1)
        
        # fc_mu2 = Conv(hidden_dims[1], hidden_dims[1], stride=1)
        # for i in range(n_blocks):
        #     fc_mu2 += [ResnetBlock(hidden_dims[1], padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False)]
        # fc_mu2 += [nn.Conv2d(hidden_dims[1], hidden_dims[1], kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)]
        # self.fc_mu2 = nn.Sequential(*fc_mu2)
        # fc_var2 = Conv(hidden_dims[1], hidden_dims[1], stride=1)
        # for i in range(n_blocks):
        #     fc_var2 += [ResnetBlock(hidden_dims[1], padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False)]
        # fc_var2 += [nn.Conv2d(hidden_dims[1], hidden_dims[1], kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)]
        # self.fc_var2 = nn.Sequential(*fc_var2)
        
        self.fc_mu0 = nn.Sequential(*Conv(in_channels, in_channels, stride=1), 
                                    nn.Conv2d(in_channels, in_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
                                    )
        self.fc_var0 = nn.Sequential(*Conv(in_channels, in_channels, stride=1), 
                                    nn.Conv2d(in_channels, in_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
                                    )
        self.fc_mu1 = nn.Sequential(*Conv(hidden_dims[0], hidden_dims[0], stride=1), 
                                    nn.Conv2d(hidden_dims[0], hidden_dims[0], kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
                                    )
        self.fc_var1 = nn.Sequential(*Conv(hidden_dims[0], hidden_dims[0], stride=1), 
                                    nn.Conv2d(hidden_dims[0], hidden_dims[0], kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
                                    )
        self.fc_mu2 = nn.Sequential(*Conv(hidden_dims[1], hidden_dims[1], stride=1), 
                                    nn.Conv2d(hidden_dims[1], hidden_dims[1], kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
                                    )
        self.fc_var2 = nn.Sequential(*Conv(hidden_dims[1], hidden_dims[1], stride=1), 
                                    nn.Conv2d(hidden_dims[1], hidden_dims[1], kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
                                    )
        
    def forward(self, input):
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

class Decoder(nn.Module):
    def __init__(self,
                in_channels: int,
                condition_channels: int, 
                n_blocks = 0) -> None:
        super(Decoder, self).__init__()

        self.in_channels = in_channels
        self.condition_channels = condition_channels

        def ConvT(input_nums, output_nums):
            layer = []
            layer.append(nn.ConvTranspose2d(input_nums, output_nums, kernel_size=(4,4), stride=(2,2), padding=(1,1), bias=False))
            layer.append(nn.BatchNorm2d(output_nums))
            layer.append(nn.ReLU())
            return layer
        
        def Conv(input_nums, output_nums, stride):
            layer = []
            layer.append(nn.Conv2d(input_nums, output_nums, kernel_size=(3,3), stride=stride, padding=(1,1), bias=False))
            layer.append(nn.BatchNorm2d(output_nums))
            layer.append(nn.ReLU())
            return layer
        
        hidden_dims = [64, 128]
        # Build Decoder

        # Layer0 = Conv(hidden_dims[1] * 2, hidden_dims[1] * 2, stride=1)
        # for i in range(n_blocks):
        #     Layer0 += [ResnetBlock(hidden_dims[1] * 2, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False)]
        # self.decoder_input0 = nn.Sequential(*Layer0)
        # Layer1 = Conv(hidden_dims[0] * 2, hidden_dims[0] * 2, stride=1)
        # for i in range(n_blocks):
        #     Layer1 += [ResnetBlock(hidden_dims[0] * 2, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False)]
        # self.decoder_input1 = nn.Sequential(*Layer1)
        # Layer2 = Conv(in_channels + condition_channels, in_channels + condition_channels, stride=1)
        # for i in range(n_blocks):
        #     Layer2 += [ResnetBlock(in_channels + condition_channels, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False)]
        # self.decoder_input2 = nn.Sequential(*Layer2)
        
        self.decoder_input0 = nn.Sequential(*Conv(hidden_dims[1] * 2, hidden_dims[1] * 2, stride=1))
        self.decoder_input1 = nn.Sequential(*Conv(hidden_dims[0] * 2, hidden_dims[0] * 2, stride=1))
        self.decoder_input2 = nn.Sequential(*Conv(in_channels + condition_channels, in_channels + condition_channels, stride=1))

        self.decconv1 = nn.Sequential(*ConvT(hidden_dims[1] * 2, hidden_dims[1]))
        self.decconv2 = nn.Sequential(*ConvT(hidden_dims[1] * 2, in_channels + condition_channels))

        self.final_layer = nn.Sequential(
                            nn.Conv2d((in_channels + condition_channels) * 2, out_channels= self.in_channels,
                                      kernel_size= 3, padding= 1, bias=False)
                            )
        self.embed_class0 = nn.Conv2d(condition_channels, condition_channels, kernel_size=1)
        self.embed_class1 = nn.Sequential(*Conv(condition_channels, hidden_dims[0], stride=2))
        self.embed_class2 = nn.Sequential(*Conv(hidden_dims[0], hidden_dims[1], stride=2))
    
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
    
    def forward(self, z0, z1, z2, y):
        embed_y0 = self.embed_class0(y)
        embed_y1 = self.embed_class1(embed_y0)
        embed_y2 = self.embed_class2(embed_y1)
        
        z0 = torch.cat([z0, embed_y2], dim = 1)
        z1 = torch.cat([z1, embed_y1], dim = 1)
        z2 = torch.cat([z2, embed_y0], dim = 1)
        
        recon = self.decode(z0, z1, z2)
        return recon

def vae_loss(recons, input, mu0, log_var0, mu1, log_var1, mu2, log_var2):

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
    return loss
