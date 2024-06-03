import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.cosine_loss = nn.CosineEmbeddingLoss()

    def forward(self, c, b):
        # Calculate the reconstruction loss (MSE between c and b)
        reconstruction_loss = self.mse_loss(c, b)

        # Compute the FFT of both c and b
        fft_c = torch.fft.fftshift(torch.fft.fft2(c))
        fft_b = torch.fft.fftshift(torch.fft.fft2(b))

        # Calculate magnitudes
        magnitude_c = torch.abs(fft_c)
        magnitude_b = torch.abs(fft_b)

        # Calculate phases
        phase_c = torch.angle(fft_c)
        phase_b = torch.angle(fft_b)

        # Compute the magnitude loss (MSE)
        magnitude_loss = self.mse_loss(magnitude_c, magnitude_b)

        # Create complex vectors for cosine similarity (cosine and sine of phases)
        vector_c = torch.stack((torch.cos(phase_c), torch.sin(phase_c)), dim=-1)  # Shape: [N, ..., 2]
        vector_b = torch.stack((torch.cos(phase_b), torch.sin(phase_b)), dim=-1)  # Shape: [N, ..., 2]

        # Flatten the vectors for embedding loss
        vector_c = vector_c.view(-1, 2)
        vector_b = vector_b.view(-1, 2)

        # Create labels (1 for similar)
        labels = torch.ones(vector_c.size(0)).to(c.device)

        # Compute the phase loss (CosineEmbeddingLoss)
        phase_loss = self.cosine_loss(vector_c, vector_b, labels)

        # Combine the reconstruction loss, magnitude loss, and phase loss
        total_loss = reconstruction_loss + magnitude_loss + phase_loss

        return total_loss



