import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np
from utils import *
import torch.nn.functional as F
from NTIRE2022Util import *

class GuidedFilter():
    def __init__(self, radius=1, eps=1e-8):
        super(GuidedFilter, self).__init__()
        self.radius = radius
        self.eps = eps

    def __call__(self, guidance, input):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        guidance = guidance.to(device)
        input = input.to(device)
        # Ensure guidance and input have the same shape
        if guidance.size() != input.size():
            raise ValueError("Guidance and input must have the same shape.")

        # Compute mean values
        mean_guidance = F.avg_pool2d(guidance, kernel_size=2*self.radius+1, stride=1, padding=self.radius)
        mean_input = F.avg_pool2d(input, kernel_size=2*self.radius+1, stride=1, padding=self.radius)

        # Compute correlation values
        correlation = F.avg_pool2d(guidance * input, kernel_size=2*self.radius+1, stride=1, padding=self.radius) - mean_guidance * mean_input

        # Compute covariance values
        mean_guidance_squared = mean_guidance**2
        mean_input_squared = mean_input**2
        covariance = F.avg_pool2d(guidance * guidance, kernel_size=2*self.radius+1, stride=1, padding=self.radius) - mean_guidance_squared + self.eps
        covariance += F.avg_pool2d(input * input, kernel_size=2*self.radius+1, stride=1, padding=self.radius) - mean_input_squared + self.eps

        # Compute linear coefficients
        a = correlation / covariance
        b = mean_input - a * mean_guidance

        # Compute mean values for a and b
        mean_a = F.avg_pool2d(a, kernel_size=2*self.radius+1, stride=1, padding=self.radius)
        mean_b = F.avg_pool2d(b, kernel_size=2*self.radius+1, stride=1, padding=self.radius)

        # Compute output
        output = mean_a * guidance + mean_b

        return output

# Example usage
if __name__ == "__main__":
    import torchvision.transforms as transforms
    from PIL import Image
    import matplotlib.pyplot as plt

    # Load an example image
    path = '/work3/s212645/Spectral_Reconstruction/clean/CAVE/003.mat'
    mat = scipy.io.loadmat(path)
    image_tensor = torch.from_numpy(mat['rgb']).unsqueeze(dim=0).permute(0, 3, 1, 2)

    # Create a guided filter instance
    guided_filter = GuidedFilter(radius=5)

    # Apply the guided filter to the image
    filtered_image = guided_filter(image_tensor, image_tensor)

    # Display the original and filtered images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(transforms.ToPILImage()(image_tensor.squeeze(0)))
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(transforms.ToPILImage()(filtered_image.squeeze(0)))
    plt.title('Filtered Image')

    plt.show()
