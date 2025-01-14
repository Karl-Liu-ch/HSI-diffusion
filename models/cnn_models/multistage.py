import torch
import torch.nn as nn
from math import sqrt
import torch.nn.functional as F


class SpectralAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpectralAttention, self).__init__()

        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )

        self.max_pool = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )

        self.sigmln = nn.Sequential(
            nn.Sigmoid(),
            nn.LayerNorm(normalized_shape=[in_channels, 1, 1])
        )

        self.beta = nn.parameter.Parameter(torch.full(size=(1,1), fill_value=0.1))

        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, input):
        avg = self.avg_pool(input)
        max = self.max_pool(input)
        conv = self.conv1x1(input)
        output = torch.add(avg, max)
        output = self.sigmln(output)
        output = output * conv
        output = torch.add(output, input*self.beta)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, in_channels, spatial_size):
        super(SpatialAttention, self).__init__()

        self.train_size = True
        self.spatial_size = spatial_size
        
        # self.norm_train = nn.LayerNorm(normalized_shape=[2, spatial_size[0], spatial_size[1]])
        # self.norm_test = nn.LayerNorm(normalized_shape=[2, 120, 128])
        self.layers = nn.Sequential(
            nn.Conv2d(2, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, input):
        b, c, h, w = input.shape
        if self.spatial_size != (h,w):
            self.train_size = False
        max_pool = torch.amax(input, dim=1, keepdim=True)
        avg_pool = input.mean(dim=1, keepdim=True)
        output = torch.cat((max_pool, avg_pool), dim=1)
        # if self.train_size: output = self.norm_train(output)
        # if not self.train_size: output = self.norm_test(output)
        output = F.layer_norm(output, [2, h, w])
        self.train_size = True
        output = self.layers(output)
        output = torch.add(input, output)
        return output

class DW1x1BNSiLU(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(DW1x1BNSiLU, self).__init__()

        if out_channels == None: # by default, the block doesn't change the size of the input
            out_channels = in_channels

        self.layers = nn.Sequential(
            DepthWiseSeperableConv3x3(in_channels, in_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.SiLU()
        )
    
    def forward(self, input):
        output = self.layers(input)
        return output


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, spatial_size):
        super(AttentionBlock, self).__init__()

        self.train_size = True
        self.in_channels = in_channels
        self.spatial_size = spatial_size
        self.spatial = SpatialAttention(in_channels, spatial_size)
        self.spectral = SpectralAttention(in_channels)

        self.layers = nn.Sequential(
            nn.Conv2d(2*in_channels, in_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
        )
        # self.norm_train = nn.LayerNorm(normalized_shape=[in_channels, spatial_size[0], spatial_size[1]])
        # self.norm_test = nn.LayerNorm(normalized_shape=[in_channels, 120, 128])

    def forward(self, input):
        b, c, h, w = input.shape
        if self.spatial_size != (h,w):
            self.train_size = False
        spa = self.spatial(input)
        spec = self.spectral(input)
        output = torch.cat((spa, spec), dim=1)
        output = self.layers(output)
        # if self.train_size: output = self.norm_train(output)
        # if not self.train_size: output = self.norm_test(output)
        output = F.layer_norm(output, [self.in_channels, h, w])
        self.train_size = True
        output = torch.add(input, output)
        return output


class EfficientSelfAttention(nn.Module):
    def __init__(self, in_channels, spatial_size):
        super(EfficientSelfAttention, self).__init__()

        self.train_size = True
        self.spatial_size = spatial_size
        self.in_channels = in_channels

        # down_spatial_train = (spatial_size[0]//2, spatial_size[1]//2)
        # down_spatial_test = (120, 128)

        self.first_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

        self.q1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        # self.q2_train = nn.AdaptiveMaxPool2d(output_size=down_spatial_train)
        # self.q2_test = nn.AdaptiveMaxPool2d(output_size=down_spatial_test)
        
        self.k1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        # self.k2_train = nn.AdaptiveMaxPool2d(output_size=down_spatial_train)
        # self.k2_test = nn.AdaptiveMaxPool2d(output_size=down_spatial_test)
        
        self.v1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        # self.v2_train = nn.AdaptiveMaxPool2d(output_size=down_spatial_train)
        # self.v2_test = nn.AdaptiveMaxPool2d(output_size=down_spatial_test)

        self.softmax = nn.Softmax(dim=1)

        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )
        # self.ffn_norm_train = nn.LayerNorm(normalized_shape=[in_channels, spatial_size[0], spatial_size[1]])
        # self.ffn_norm_test = nn.LayerNorm(normalized_shape=[in_channels, 240, 256])
        self.ffn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )

    def forward(self, input):
        b, c, h, w = input.shape
        if self.spatial_size != (h,w):
            self.train_size = False

        output = self.first_conv(input)

        # if self.train_size:
        Q = F.adaptive_max_pool2d(self.q1(output), output_size=(h // 2, w // 2))
        K = F.adaptive_max_pool2d(self.k1(output), output_size=(h // 2, w // 2))
        V = F.adaptive_max_pool2d(self.v1(output), output_size=(h // 2, w // 2))
        
        self.scale_train = 1/sqrt((h + w)/2)
        # self.scale_test = 1/sqrt((down_spatial_test[0]+down_spatial_test[1])/2)

        output = Q @ K.transpose(-2, -1) * self.scale_train
        # else:
        #     Q = self.q2_test(self.q1(output))
        #     K = self.k2_test(self.k1(output))
        #     V = self.v2_test(self.v1(output))
        #     output = Q @ K.transpose(-2, -1) * self.scale_test
        
        output = self.softmax(output)
        output = output @ V
        output = self.up(output)
        output = torch.add(input, output)
        # if self.train_size: output = self.ffn_norm_train(output)
        # if not self.train_size: output = self.ffn_norm_test(output)
        output = F.layer_norm(output, [self.in_channels, h, w])
        output = self.ffn(output)
        self.train_size = True
        return output


class PixelUnshuffleBlock(nn.Module):
    """
    A block composed of a pixel unshuffle layer followed by a convolution to control 
    the number of channels

    input : C*H*W
    output : C*H/2*W/2
    """
    def __init__(self, in_channels, out_channels=None, downscale_factor=2):
        super(PixelUnshuffleBlock, self).__init__()
        if out_channels == None: # by default, the block doesn't change the size of the input
            out_channels = in_channels
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor)
        self.conv1x1 = nn.Conv2d(in_channels*(downscale_factor**2), out_channels, kernel_size=1)

    
    def forward(self, input):
        output = self.pixel_unshuffle(input)
        output = self.conv1x1(output)
        return output


class PixelShuffleBlock(nn.Module):
    """
    A block composed of a pixel shuffle layer followed by a convolution to control the number of channels
    There is also a convolution before the pixel shuffle to make sure the number of channels is divisible by upscale_factor**2

    input : C*H*W
    output : C*2H*2W
    """
    def __init__(self, in_channels, out_channels=None, upscale_factor=2):
        super(PixelShuffleBlock, self).__init__()
        if out_channels == None: # by default, the block doesn't change the number of channels of the input
            out_channels = in_channels
        if in_channels % (upscale_factor**2) != 0:
            shuffle_channels = ((in_channels//(upscale_factor**2)) + 1) * (upscale_factor**2)
        else: shuffle_channels = in_channels

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, shuffle_channels, kernel_size=1),
            nn.PixelShuffle(upscale_factor),
            nn.Conv2d(int(shuffle_channels/(upscale_factor**2)), out_channels, kernel_size=1)
        )


    def forward(self, input):
        output = self.layers(input)
        return output


class DepthWiseSeperableConv3x3(nn.Module):
    """
    This block defines a depth-wise separable conv with a kernel size of 3
    """
    def __init__(self, in_channels, out_channels):
        super(DepthWiseSeperableConv3x3, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)


    def forward(self, input):
        output = self.depthwise(input)
        output = self.pointwise(output)
        return output



class Multistage(nn.Module):

    def __init__(self, input_spatial_size):
        super(Multistage, self).__init__()

        self.input_spatial_size = input_spatial_size
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            DW1x1BNSiLU(in_channels=64)
        )

        self.stage2 = nn.Sequential(
            PixelUnshuffleBlock(in_channels=64, out_channels=128),
            DW1x1BNSiLU(in_channels=128),
            DW1x1BNSiLU(in_channels=128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(num_features=128),
            nn.SiLU()
        )

        self.stage3 = nn.Sequential(
            PixelUnshuffleBlock(in_channels=128, out_channels=256),
            AttentionBlock(in_channels=256, spatial_size=input_spatial_size)
        )

        self.stage4 = nn.Sequential(
            PixelShuffleBlock(in_channels=256, out_channels=128),
            DW1x1BNSiLU(in_channels=128),
            EfficientSelfAttention(in_channels=128, spatial_size=input_spatial_size),
            PixelShuffleBlock(in_channels=128, out_channels=64),
            nn.Conv2d(in_channels=64, out_channels=31, kernel_size=3, padding=1)
        )

    def forward(self, input):

        b, c, h_inp, w_inp = input.shape
        pad_h = (self.input_spatial_size - h_inp % self.input_spatial_size) % self.input_spatial_size
        pad_w = (self.input_spatial_size - w_inp % self.input_spatial_size) % self.input_spatial_size
        input = F.pad(input, [0, pad_w, 0, pad_h], mode='reflect')

        output = self.stage1(input)
        output = self.stage2(output)
        output = self.stage3(output)
        output = self.stage4(output)
        return output[:, :, :h_inp, :w_inp]
    

if __name__ == '__main__':
    model = Multistage(128)
    input = torch.rand(1,3,512,482)
    output = model(input)
    print(output.shape)