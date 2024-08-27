import torch
import torch.nn as nn
import torch.fft

class KKLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(KKLayer, self).__init__()
        # Each output channel has its own set of alpha and beta parameters
        self.alphas = nn.Parameter(torch.randn(out_channels, in_channels))
        self.betas = nn.Parameter(torch.randn(out_channels, in_channels))

    def forward(self, x):
        batch_size, in_channels, height, width = x.shape

        # Fourier Transform
        x_fft = torch.fft.fft2(x)

        # Initialize an output tensor
        output = torch.zeros(batch_size, self.alphas.shape[0], height, width, dtype=torch.float32).to(x.device)

        for i in range(self.alphas.shape[0]):  # Iterate over output channels
            new_real = torch.zeros_like(x_fft.real)
            new_imag = torch.zeros_like(x_fft.imag)

            for j in range(in_channels):  # Iterate over input channels
                real_part = x_fft[:, j, :, :].real
                imag_part = x_fft[:, j, :, :].imag
                
                # Apply parameterized transformation
                new_real[:, j, :, :] = self.alphas[i, j] * real_part + self.betas[i, j] * imag_part
                new_imag[:, j, :, :] = self.betas[i, j] * real_part - self.alphas[i, j] * imag_part
            
            # Reconstruct the complex feature map
            x_fft_modified = torch.complex(new_real.sum(dim=1), new_imag.sum(dim=1))

            # Inverse Fourier Transform
            x_ifft = torch.fft.ifft2(x_fft_modified)
            output[:, i, :, :] = torch.real(x_ifft)  # Keep only the real part

        return output
