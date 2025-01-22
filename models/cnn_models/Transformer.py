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
        self.ln_train = nn.LayerNorm([out_channels, spatial_size[0], spatial_size[1]])  # LayerNorm on feature map shape
        self.ln_test = nn.LayerNorm([out_channels, spatial_size_test[0], spatial_size_test[1]])
        self.out_channels  = out_channels
    
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
        attention_scores = attention_scores/scaling  # Apply scaling
        
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


class SkyformerSpatialAttention(nn.Module):
    def __init__(self, dim, num_heads=4, num_landmarks=64, kernel_size=3, init_sigma=0.5, regularization=1e-6):
        """
        Skyformer-based spatial attention with kernelized Gaussian attention and improved landmark sampling.
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            num_landmarks (int): Number of Nyström landmarks for attention approximation.
            kernel_size (int): Size of convolution kernel for feature extraction.
            init_sigma (float): Initial value for learnable Gaussian kernel's sigma.
            regularization (float): Regularization term for kernel inversion.
        """
        super(SkyformerSpatialAttention, self).__init__()
        self.num_heads = num_heads
        self.num_landmarks = num_landmarks
        self.regularization = regularization
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
 
        # Feature projections
        self.query_proj = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=num_heads)
        self.key_proj = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=num_heads)
        self.value_proj = nn.Conv2d(dim, dim, kernel_size=1)
 
        # Learnable sigma for Gaussian kernel
        self.sigma = nn.Parameter(torch.full((num_heads, 1, 1), init_sigma))
 
        # Output projection
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)
 
    def kernel_function(self, Q, K):
        """
        Apply Gaussian kernel to Query and Key tensors.
        Args:
            Q (Tensor): Query tensor of shape (B, num_heads, N, head_dim).
            K (Tensor): Key tensor of shape (B, num_heads, N, head_dim).
        Returns:
            Tensor: Kernelized attention matrix of shape (B, num_heads, N, L).
        """
        Q_norm = torch.sum(Q**2, dim=-1, keepdim=True)  # (B, num_heads, N, 1)
        K_norm = torch.sum(K**2, dim=-1, keepdim=True)  # (B, num_heads, 1, L)
        QK = torch.einsum("bhnd,bhld->bhnl", Q, K)  # (B, num_heads, N, L)
 
        kernel = torch.exp((2 * QK - Q_norm - K_norm.transpose(-2, -1)) / (2 * self.sigma**2))
        return kernel
 
    def structured_landmark_sampling(self, K, H, W):
        """
        Structured sampling of landmarks based on grid-based spatial partitioning.
        Args:
            K (Tensor): Key tensor of shape (B, num_heads, N, head_dim).
            H (int): Height of the input feature map.
            W (int): Width of the input feature map.
        Returns:
            Tensor: Selected landmarks of shape (B, num_heads, L, head_dim).
        """
        B, num_heads, N, head_dim = K.shape
        grid_h = torch.linspace(0, H - 1, int(self.num_landmarks ** 0.5), device=K.device).long()
        grid_w = torch.linspace(0, W - 1, int(self.num_landmarks ** 0.5), device=K.device).long()
       
        # Compute grid indices
        grid_indices = torch.cartesian_prod(grid_h, grid_w).to(K.device)
        grid_indices = grid_indices[:, 0] * W + grid_indices[:, 1]  # Map 2D indices to 1D
 
        # Handle cases where the number of grid indices exceeds landmarks
        selected_indices = grid_indices[:self.num_landmarks]
 
        return K[:, :, selected_indices, :]
 
    def forward(self, x):
        """
        Forward pass for spatial attention.
        Args:
            x (Tensor): Input feature map of shape (B, dim, H, W).
        Returns:
            Tensor: Output feature map with spatial attention applied.
        """
        B, dim, H, W = x.size()
        N = H * W
 
        # Project to Q, K, V
        Q = self.query_proj(x).view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)  # (B, num_heads, N, head_dim)
        K = self.key_proj(x).view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)  # (B, num_heads, N, head_dim)
        V = self.value_proj(x).view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)  # (B, num_heads, N, head_dim)
 
        # Structured landmark sampling
        K_landmarks = self.structured_landmark_sampling(K, H, W)  # (B, num_heads, L, head_dim)
        V_landmarks = self.structured_landmark_sampling(V, H, W)  # (B, num_heads, L, head_dim)
 
        # Kernelized attention
        K_QK = self.kernel_function(Q, K_landmarks)  # (B, num_heads, N, L)
        K_landmarks_K = self.kernel_function(K_landmarks, K_landmarks)  # (B, num_heads, L, L)
 
        # Regularized inversion of landmark kernel matrix
        K_landmarks_K = K_landmarks_K + self.regularization * torch.eye(self.num_landmarks, device=Q.device).unsqueeze(0).unsqueeze(0)
        K_landmarks_K_inv = torch.linalg.pinv(K_landmarks_K)  # (B, num_heads, L, L)
 
        # Normalize kernel and compute attention output
        K_QK_normalized = torch.matmul(K_QK, K_landmarks_K_inv)  # (B, num_heads, N, L)
        attention_result = torch.matmul(K_QK_normalized, V_landmarks)  # (B, num_heads, N, head_dim)
 
        # Reshape and project back
        attention_result = attention_result.permute(0, 1, 3, 2).reshape(B, dim, H, W)
        out = self.out_proj(attention_result)  # (B, dim, H, W)
 
        return out



class SpatiallyGatedFFN(nn.Module):
    def __init__(self, dim, expansion_factor=4, kernel_size=3, dropout=0.1):
        """
        Spatially Gated Feed-Forward Network (SG-FFN).
        Args:
            dim (int): Number of input and output channels.
            expansion_factor (int): Expansion factor for hidden dimension.
            kernel_size (int): Size of the spatial kernel for gating.
            dropout (float): Dropout rate.
        """
        super(SpatiallyGatedFFN, self).__init__()
        hidden_dim = dim * expansion_factor
 
        # Pointwise expansion
        self.expand = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            nn.GELU()
        )
 
        # Spatial gating
        self.gate = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=hidden_dim),
            nn.Sigmoid(),
            nn.Dropout(p=dropout)
        )
 
        # Pointwise compression
        self.compress = nn.Conv2d(hidden_dim, dim, kernel_size=1)
 
        # LayerNorm applied along channel dimension
        self.norm = nn.LayerNorm(dim)
 
    def forward(self, x):
        """
        Forward pass for SG-FFN.
        Args:
            x (Tensor): Input tensor of shape (B, dim, H, W).
        Returns:
            Tensor: Output tensor of shape (B, dim, H, W).
        """
        residual = x
        B, C, H, W = x.shape
 
        # Expand and gate
        x = self.expand(x)  # (B, hidden_dim, H, W)
        gated_x = self.gate(x)  # (B, hidden_dim, H, W)
 
        # Compress
        x = self.compress(gated_x)  # (B, dim, H, W)
 
        # Normalize (apply LayerNorm per channel)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.norm(x)  # (B, H, W, C)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
 
        # Add residual connection
        return x + residual


class DualTransformerBlockSpaSpec(nn.Module):
    def __init__(self, in_channels, spatial_size, wl_decomp, out_channels=None):
        super().__init__()

        self.train_size = True
        self.spatial_size = spatial_size

        if spatial_size[0] == 128: spatial_size_test = (480, 512)
        if spatial_size[0] == 64: spatial_size_test = (240, 256)
        if spatial_size[0] == 32: spatial_size_test = (120, 128)

        self.ln1_train = nn.LayerNorm([in_channels, spatial_size[0], spatial_size[1]])
        self.ln1_test = nn.LayerNorm([in_channels, spatial_size_test[0], spatial_size_test[1]])

        self.spa_att = SkyformerSpatialAttention(in_channels)
            
        self.ln2_train = nn.LayerNorm([in_channels, spatial_size[0], spatial_size[1]])
        self.ln2_test = nn.LayerNorm([in_channels, spatial_size_test[0], spatial_size_test[1]])

        self.spa_ffn = SpatiallyGatedFFN(in_channels)

        self.ln3_train = nn.LayerNorm([in_channels, spatial_size[0], spatial_size[1]])
        self.ln3_test = nn.LayerNorm([in_channels, spatial_size_test[0], spatial_size_test[1]])

        self.spec_att = SparsemaxAttentionWithLearnableScaling(in_channels, spatial_size, wl_decomp)

        self.ln4_train = nn.LayerNorm([in_channels, spatial_size[0], spatial_size[1]])
        self.ln4_test = nn.LayerNorm([in_channels, spatial_size_test[0], spatial_size_test[1]])
        self.in_channels = in_channels

        self.spec_ffn = DynamicallyGatedSpectralFFN(in_channels)
    
    def forward(self, input):
        b, c, h, w = input.shape
        if self.spatial_size != (h,w):
            self.train_size = False
        mid = F.layer_norm(input, [self.in_channels, h, w])
        output = F.layer_norm(torch.add(self.spa_att(mid), input), [self.in_channels, h, w])
        output = torch.add(self.spa_ffn(output), output)
        # if self.train_size: output = self.ln3_train(output)
        # if not self.train_size: output = self.ln3_test(output)
        output = F.layer_norm(output, [self.in_channels, h, w])
        specatt, attention_norm = self.spec_att(output)
        output = torch.add(specatt, output)
        # if self.train_size: output = torch.add(self.spec_ffn(self.ln4_train(output)), output)
        # if not self.train_size: output = torch.add(self.spec_ffn(self.ln4_test(output)), output)
        output = torch.add(self.spec_ffn(F.layer_norm(output,[self.in_channels, h, w])), output)
        self.train_size = True
        return output, attention_norm


class DualTransformerBlockSpa(nn.Module):
    def __init__(self, in_channels, spatial_size, wl_decomp, out_channels=None):
        super().__init__()

        self.train_size = True
        self.spatial_size = spatial_size

        if spatial_size[0] == 128: spatial_size_test = (480, 512)
        if spatial_size[0] == 64: spatial_size_test = (240, 256)
        if spatial_size[0] == 32: spatial_size_test = (120, 128)

        self.ln1_train = nn.LayerNorm([in_channels, spatial_size[0], spatial_size[1]])
        self.ln1_test = nn.LayerNorm([in_channels, spatial_size_test[0], spatial_size_test[1]])

        self.spa_att = SkyformerSpatialAttention(in_channels)

        self.ln2_train = nn.LayerNorm([in_channels, spatial_size[0], spatial_size[1]])
        self.ln2_test = nn.LayerNorm([in_channels, spatial_size_test[0], spatial_size_test[1]])
        self.in_channels = in_channels
        
        self.spa_ffn = SpatiallyGatedFFN(in_channels)
    
    def forward(self, input):
        b, c, h, w = input.shape
        if self.spatial_size != (h,w):
            self.train_size = False
        
        mid = F.layer_norm(input, [self.in_channels, h, w])
        output = F.layer_norm(torch.add(self.spa_att(mid), input), [self.in_channels, h, w])

        # if self.train_size: output = self.ln2_train(torch.add(self.spa_att(self.ln1_train(input)), input))
        # if not self.train_size: output = self.ln2_test(torch.add(self.spa_att(self.ln1_test(input)), input))
        output = torch.add(self.spa_ffn(output), output)
        self.train_size = True
        return output


class DualTransformerBlockSpec(nn.Module):
    def __init__(self, in_channels, spatial_size, wl_decomp, out_channels=None):
        super().__init__()

        self.train_size = True
        self.spatial_size = spatial_size

        if spatial_size[0] == 128: spatial_size_test = (480, 512)
        if spatial_size[0] == 64: spatial_size_test = (240, 256)
        if spatial_size[0] == 32: spatial_size_test = (120, 128)

        self.ln_train = nn.LayerNorm([in_channels, spatial_size[0], spatial_size[1]])
        self.ln_test = nn.LayerNorm([in_channels, spatial_size_test[0], spatial_size_test[1]])

        self.spec_att = SparsemaxAttentionWithLearnableScaling(in_channels, spatial_size, wl_decomp)

        self.spec_ffn_norm_train = nn.LayerNorm([in_channels, spatial_size[0], spatial_size[1]])
        self.spec_ffn_norm_test = nn.LayerNorm([in_channels, spatial_size_test[0], spatial_size_test[1]])
        self.in_channels = in_channels
        
        self.spec_ffn = DynamicallyGatedSpectralFFN(in_channels)
    
    def forward(self, input):
        b, c, h, w = input.shape
        if self.spatial_size != (h,w):
            self.train_size = False
        # if self.train_size: output = self.ln_train(input)
        # if not self.train_size: output = self.ln_test(input)
        
        output = F.layer_norm(input, [self.in_channels, h, w])

        specatt, attention_norm = self.spec_att(output)
        output = torch.add(specatt, output)
        output = torch.add(self.spec_ffn(F.layer_norm(output, [self.in_channels, h, w])), output)
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
    def __init__(self):
        super().__init__()

        self.input_stage = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            DualTransformerBlockSpaSpec(in_channels=64, spatial_size=(128, 128), wl_decomp=1)
        )
        self.midlevel_stage = nn.Sequential(
            PixelUnshuffleBlock(in_channels=64, out_channels=128),
            DualTransformerBlockSpaSpec(in_channels=128, spatial_size=(64,64), wl_decomp=2)
        )
        self.deep_stage = nn.Sequential(
            PixelUnshuffleBlock(in_channels=128, out_channels=256),
            DualTransformerBlockSpaSpec(in_channels=256, spatial_size=(32,32), wl_decomp=3)
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
        return output[:, :, :h_inp, :w_inp]

        
if __name__ == '__main__':
    model = Transformer()
    input = torch.rand(1,3,512,482)
    output = model(input)
    print(output.shape)