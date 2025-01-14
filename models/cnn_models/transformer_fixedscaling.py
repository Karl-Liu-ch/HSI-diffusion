import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward, DWTInverse
from sparsemax import Sparsemax
from math import log, sqrt
import torch.nn.functional as F



class ProjectionBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, spatial_size, out_channels=None):
        super(ProjectionBlock, self).__init__()

        if out_channels==None: # by default the block doesn't change the size of the input
            out_channels = in_channels
        
        self.train_size = True
        self.spatial_size = spatial_size
        
        if spatial_size[0] == 64: spatial_size_test = (240, 256)
        elif spatial_size[0] == 16: spatial_size_test = (60, 64)
        elif spatial_size[0] == 4: spatial_size_test = (15, 16)

        # 1x1 convolution -> Depthwise convolution -> LayerNorm
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.dwconv = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, groups=out_channels, padding=kernel_size // 2)  # Depthwise convolution
        # self.ln_train = nn.LayerNorm([out_channels, spatial_size[0], spatial_size[1]])  # LayerNorm on feature map shape
        # self.ln_test = nn.LayerNorm([out_channels, spatial_size_test[0], spatial_size_test[1]])
        self.out_channels = out_channels
    
    def forward(self, x):
        b, c, h, w = x.shape
        if self.spatial_size != (h,w):
            self.train_size = False
        x = self.conv1x1(x)  # Apply 1x1 convolution
        x = self.dwconv(x)   # Apply depthwise convolution
        # Apply LayerNorm
        # if self.train_size: x = self.ln_train(x)
        # if not self.train_size: x = self.ln_test(x)
        x = F.layer_norm(x, [self.out_channels, h, w])
        self.train_size = True
        return x


class OldPositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super(OldPositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * -(log(10000.0) / embed_size))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # Assuming x has shape [B, C, H, W] and we are adding position to the C dimension
        seq_len = x.size(2) * x.size(3)  # H * W, flattened spatial size
        return self.pe[:, :seq_len]


class PositionalEncoding(nn.Module):
    def __init__(self, in_channels):
        super(PositionalEncoding, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        )
    
    def forward(self, input):
        output = self.layers(input)
        return output



class SparsemaxAttentionWithLearnableScaling(nn.Module):
    def __init__(self, in_channels, spatial_size, wl_decomp, out_channels=None):
        super(SparsemaxAttentionWithLearnableScaling, self).__init__()

        if out_channels==None: # by default the block doesn't change the size of the input
            out_channels = in_channels

        self.num_heads = 8 # number of heads for multi-head attention

        self.positional_encoding = PositionalEncoding(in_channels)

        self.wavelet = DWTForward(J=wl_decomp, wave='haar')
        self.inverse_wavelet = DWTInverse(wave='haar')
        
        # Projection blocks
        if spatial_size[0] == 128: spatial_size = (64,64)
        elif spatial_size[0] == 64: spatial_size = (16,16)
        elif spatial_size[0] == 32: spatial_size = (4,4)

        self.query_proj = ProjectionBlock(in_channels, kernel_size=5, spatial_size=spatial_size)  # 1x1 -> DW5x5 -> LN
        self.key_proj = ProjectionBlock(in_channels, kernel_size=3, spatial_size=spatial_size)    # 1x1 -> DW3x3 -> LN
        self.value_proj = ProjectionBlock(in_channels, kernel_size=3, spatial_size=spatial_size)  # 1x1 -> DW3x3 -> LN
        
        # Learnable scaling factor
        #self.scale_factor = nn.Parameter(torch.tensor(1/spatial_size))  # Initialize scaling factor to 1
        self.scale_factor = sqrt(in_channels)
        
        # Sparsemax attention
        self.sparsemax = Sparsemax(dim=-1)

        # 1x1 conv after concatenation of the heads
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
    
    def normalize(self, x):
        """Normalize the input tensor along the feature dimension."""
        return (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6)  # Prevent division by zero
    
    def forward(self, x):
        # Step 1: Add Positional Encoding
        position = self.positional_encoding(x)
        x = x + position  # Element-wise addition of positional encoding to the input
        
        # Step 2: Apply wavelet transform
        wavelet_coeffs, wavelet_yh = self.wavelet(x)
        
        # Step 3: Projection to queries, keys, and values using the specified projections
        Q = self.query_proj(wavelet_coeffs)  # Shape [B, Q, H, W]
        K = self.key_proj(wavelet_coeffs)    # Shape [B, K, H, W]
        V = self.value_proj(wavelet_coeffs)  # Shape [B, V, H, W]
        
        # Step 4: Normalize query, key, and value
        Q = self.normalize(Q)  # Normalize queries
        K = self.normalize(K)  # Normalize keys
        V = self.normalize(V)  # Normalize values
        
        # Step 5: Reshape to [B, N, H * W] where N is the number of features
        B, C, H, W = Q.shape
        Q = Q.view(B, C, -1)  # Flatten spatial dimensions
        K = K.view(B, C, -1)  # Flatten spatial dimensions
        V = V.view(B, C, -1)  # Flatten spatial dimensions

        # split the attention matrix to perform multi-head self attention
        # dim after transformation : [B, num_heads, C, H*W]
        Q = Q.view(B, C, self.num_heads, -1).transpose(1, 2)
        K = K.view(B, C, self.num_heads, -1).transpose(1, 2)
        V = V.view(B, C, self.num_heads, -1).transpose(1, 2)

        # Step 6: Scaled dot product attention with learnable scale factor
        scaling = self.scale_factor  # Use the learnable scaling factor
        attention_scores = Q @ K.transpose(2, 3)  # Dot product Q * K^T
        attention_scores /= scaling  # Apply scaling
        
        # Step 7: Apply sparsemax to attention scores
        attention_weights = self.sparsemax(attention_scores)  # Sparsemax on the scaled dot product

        # Step 8: Apply attention to values
        attention_output = attention_weights @ V

        # concatenate the heads
        heads = [attention_output[:, head, :, :] for head in range(self.num_heads)]
        attention_output = torch.cat(heads, dim=2)
        
        # Step 9: Reshape back to [B, C, H, W]
        attention_output = attention_output.view(B, C, H, W)

        # apply 1x1 conv
        attention_output = self.conv1x1(attention_output)
        
        # Step 10: Inverse wavelet transform to return to original space
        output = self.inverse_wavelet((attention_output, wavelet_yh))

        attention_norm = torch.abs(attention_weights).sum() / B
        
        return output, attention_norm


class DynamicallyGatedSpectralFFN(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()

        if out_channels==None: # by default the block doesn't change the size of the input
            out_channels = in_channels

        self.hbranch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )

        self.gbranch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        h = self.hbranch(input)
        g = self.gbranch(input)
        output = h * g
        output = torch.add(output, input)
        return output


class DualTransformerBlock(nn.Module):
    def __init__(self, in_channels, spatial_size, wl_decomp, out_channels=None):
        super().__init__()

        self.train_size = True
        self.spatial_size = spatial_size
        self.in_channels = in_channels

        if spatial_size[0] == 128: spatial_size_test = (480, 512)
        if spatial_size[0] == 64: spatial_size_test = (240, 256)
        if spatial_size[0] == 32: spatial_size_test = (120, 128)

        self.spec_att = SparsemaxAttentionWithLearnableScaling(in_channels, spatial_size, wl_decomp)

        # self.spec_ffn_norm_train = nn.LayerNorm([in_channels, spatial_size[0], spatial_size[1]])
        # self.spec_ffn_norm_test = nn.LayerNorm([in_channels, spatial_size_test[0], spatial_size_test[1]])
        
        self.spec_ffn = DynamicallyGatedSpectralFFN(in_channels)
    
    def forward(self, input):
        b, c, h, w = input.shape
        if self.spatial_size != (h,w):
            self.train_size = False
        specatt, attention_norm = self.spec_att(input)
        output = torch.add(specatt, input)
        # if self.train_size: output = torch.add(self.spec_ffn(self.spec_ffn_norm_train(output)), output)
        # if not self.train_size: output = torch.add(self.spec_ffn(self.spec_ffn_norm_test(output)), output)
        output = F.layer_norm(output, [self.in_channels, h, w])
        self.train_size = True
        return output, attention_norm


class FeatureRefinement(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.local_refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=31, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(in_channels=31, out_channels=31, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.conv1x1_1 = nn.Conv2d(in_channels, out_channels=31, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(in_channels, out_channels=31, kernel_size=1)
    
    def forward(self, input):
        output = self.local_refine(input)
        output = self.conv1x1_1(output) * self.spatial_refine(output)
        output = torch.add(output, self.conv1x1_2(input))
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


class Transformer(nn.Module):
    def __init__(self, input_spatial_size = 128):
        super().__init__()

        self.input_stage = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            DualTransformerBlock(in_channels=64, spatial_size=(128, 128), wl_decomp=1)
        )
        self.midlevel_stage = nn.Sequential(
            PixelUnshuffleBlock(in_channels=64, out_channels=128),
            DualTransformerBlock(in_channels=128, spatial_size=(64,64), wl_decomp=2)
        )
        self.deep_stage = nn.Sequential(
            PixelUnshuffleBlock(in_channels=128, out_channels=256),
            DualTransformerBlock(in_channels=256, spatial_size=(32,32), wl_decomp=3)
        )
        self.residual_connection = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            PixelUnshuffleBlock(in_channels=64, out_channels=256, downscale_factor=4)
        )
        self.refinement = nn.Sequential(
            PixelShuffleBlock(in_channels=256, out_channels=64, upscale_factor=4),
            FeatureRefinement(in_channels=64)
        )
        self.output_stage = nn.Conv2d(in_channels=31, out_channels=31, kernel_size=1)

    def forward(self, input):
        self.input_spatial_size = 128
        b, c, h_inp, w_inp = input.shape
        pad_h = (self.input_spatial_size - h_inp % self.input_spatial_size) % self.input_spatial_size
        pad_w = (self.input_spatial_size - w_inp % self.input_spatial_size) % self.input_spatial_size
        input = F.pad(input, [0, pad_w, 0, pad_h], mode='reflect')

        output, attention_norm1 = self.input_stage(input)
        output, attention_norm2 = self.midlevel_stage(output)
        output, attention_norm3 = self.deep_stage(output)
        output = torch.add(output, self.residual_connection(input))
        output = self.refinement(output)
        output = self.output_stage(output)
        attention_norm = (attention_norm1 + attention_norm2 + attention_norm3)/3
        return output[:, :, :h_inp, :w_inp], attention_norm
    

if __name__ == '__main__':
    model = Transformer()
    input = torch.rand(1,3,512,482)
    output = model(input)
    print(output[0].shape)