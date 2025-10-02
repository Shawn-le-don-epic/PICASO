import torch
from torch import nn

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder: Takes an image and compresses it.
        # Input: 3 channels (RGB), 128x128 pixels
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=2),  # -> 16 channels, 64x64
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2), # -> 32 channels, 32x32
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2), # -> 64 channels, 16x16 (This is our compressed bottleneck)
            nn.ReLU()
        )
        
        # Decoder: Takes the compressed data and reconstructs the image.
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # -> 32 channels, 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # -> 16 channels, 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> 3 channels, 128x128
            nn.Sigmoid() # Use Sigmoid to ensure output values are between 0 and 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# In model.py

import torch
from torch import nn

# (The original Autoencoder class can stay here if you want)

class AutoencoderWithSkip(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder layers are the same
        self.enc1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1, stride=2), nn.ReLU())   # -> 64x64
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.ReLU())  # -> 32x32
        self.enc3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1, stride=2), nn.ReLU()) # -> 16x16 (Bottleneck)

        # Decoder layers are the same
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU()) # -> 32x32
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(64 * 2, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU()) # -> 64x64
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(32 * 2, 3, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid())  # -> 128x128

    # --- NEW: An explicit encode method ---
    def encode(self, x):
        # Pass through encoder layers
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        bottleneck = self.enc3(e2)
        # Return the final compressed data AND the skip connections
        return bottleneck, e2, e1

    # --- NEW: An explicit decode method ---
    def decode(self, bottleneck, e2, e1):
        # Pass through decoder layers, using the skip connections
        d1 = self.dec1(bottleneck)
        d1_with_skip = torch.cat([d1, e2], dim=1) 
        
        d2 = self.dec2(d1_with_skip)
        d2_with_skip = torch.cat([d2, e1], dim=1)
        
        reconstructed = self.dec3(d2_with_skip)
        return reconstructed

    def forward(self, x):
        # The forward pass is now just a combination of encode and decode
        # This is used for training
        bottleneck, e2, e1 = self.encode(x)
        return self.decode(bottleneck, e2, e1)

'''    
class AutoencoderWithSkip(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1, stride=2), nn.ReLU())   # -> 64x64
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.ReLU())  # -> 32x32
        self.enc3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1, stride=2), nn.ReLU()) # -> 16x16 (Bottleneck)

        # Decoder
        # The input channels are doubled because we will concatenate the skip connection
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU()) # -> 32x32
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(64 * 2, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU()) # -> 64x64
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(32 * 2, 3, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid())  # -> 128x128

    def forward(self, x):
        # --- Encoder with skip connection outputs ---
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        bottleneck = self.enc3(e2)

        # --- Decoder with skip connection inputs ---
        d1 = self.dec1(bottleneck)
        # Concatenate (join) the output with the skip connection from the encoder
        d1_with_skip = torch.cat([d1, e2], dim=1) 

        d2 = self.dec2(d1_with_skip)
        d2_with_skip = torch.cat([d2, e1], dim=1)

        reconstructed = self.dec3(d2_with_skip)

        return reconstructed
'''