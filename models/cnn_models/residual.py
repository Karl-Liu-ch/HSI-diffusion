import torch
import torch.nn as nn


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
        self.norm_train = nn.LayerNorm(normalized_shape=[2, spatial_size[0], spatial_size[1]])
        self.norm_test = nn.LayerNorm(normalized_shape=[2, 60, 64])
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
        if self.train_size: output = self.norm_train(output)
        if not self.train_size: output = self.norm_test(output)
        self.train_size = True
        output = self.layers(output)
        output = torch.add(input, output)
        return output


class DFEBlock(nn.Module):
    def __init__(self, in_channels):
        super(DFEBlock, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=2, padding=2),
            nn.BatchNorm2d(in_channels),
            nn.SiLU()
        )
        self.layer3 = nn.Sequential(
            DepthWiseSeperableConv(2*in_channels, in_channels, kernel_size=3, padding=1),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )

    def forward(self, input):
        branch1 = self.layer1(input)
        branch2 = self.layer2(input)
        output = torch.cat((branch1, branch2), dim=1)
        output = self.layer3(output)
        return output


class RefineBlock(nn.Module):
    def __init__(self, in_channels):
        super(RefineBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU()
        )

    def forward(self, input):
        output = self.layers(input)
        return output



class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()

        self.dfe = DFEBlock(in_channels)
        self.refine = RefineBlock(in_channels)

    def forward(self, input):
        output = self.dfe(input)
        output = torch.add(input, output)
        output = self.refine(output)
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
        self.norm_train = nn.LayerNorm(normalized_shape=[in_channels, spatial_size[0], spatial_size[1]])
        self.norm_test = nn.LayerNorm(normalized_shape=[in_channels, 60, 64])

    def forward(self, input):
        b, c, h, w = input.shape
        if self.spatial_size != (h,w):
            self.train_size = False
        spa = self.spatial(input)
        spec = self.spectral(input)
        output = torch.cat((spa, spec), dim=1)
        output = self.layers(output)
        if self.train_size: output = self.norm_train(output)
        if not self.train_size: output = self.norm_test(output)
        self.train_size = True
        output = torch.add(input, output)
        return output


class SFEBlock(nn.Module):
    def __init__(self, in_channels):
        super(SFEBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU()
        )

    def forward(self, input):
        output = self.layers(input)
        return output


class PixelUnshuffleBlock(nn.Module):
    """
    A block composed of a pixel unshuffle layer followed by a 1x1 convolution to control 
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


class DepthWiseSeperableConv(nn.Module):
    """
    This block defines a depth-wise separable conv
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(DepthWiseSeperableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, input):
        output = self.depthwise(input)
        output = self.pointwise(output)
        return output


class Residual(nn.Module):
    def __init__(self, input_spatial_size):
        super(Residual, self).__init__()

        self.input_spatial_size = input_spatial_size
        self.first_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1),
            SFEBlock(in_channels=3),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1),
            SFEBlock(in_channels=64),
            PixelUnshuffleBlock(in_channels=64, out_channels=128),
            DFEBlock(in_channels=128),
            PixelUnshuffleBlock(in_channels=128, out_channels=256),
            DFEBlock(in_channels=256),
            PixelUnshuffleBlock(in_channels=256, out_channels=512),
            AttentionBlock(in_channels=512, spatial_size=(input_spatial_size[0]//8, input_spatial_size[1]//8)),
            ResidualBlock(in_channels=512),
            ResidualBlock(in_channels=512),
            ResidualBlock(in_channels=512),
            PixelShuffleBlock(in_channels=512, out_channels=256),
            PixelShuffleBlock(in_channels=256, out_channels=128),
            PixelShuffleBlock(in_channels=128, out_channels=62),
            DFEBlock(in_channels=62)
        )

        self.rescon = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=62, kernel_size=1),
            nn.Conv2d(in_channels=62, out_channels=62, kernel_size=3, padding=1)
        )
    
        self.last_layers = nn.Sequential(
            nn.Conv2d(in_channels=62, out_channels=31, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=31, out_channels=31, kernel_size=1)
        )
    
    def forward(self, input):
        output = self.first_layers(input)
        input = self.rescon(input)
        output = torch.add(input, output)
        output = self.last_layers(output)
        return output
    
