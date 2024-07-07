#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 11:26:16 2024

@author: hxkhkh

AUDIO PROCESSING UNET EXAAMPLE

"""

import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Define the encoder path
        self.enc1 = self.conv_block(1, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Define the decoder path
        self.dec4 = self.up_conv_block(1024, 512)
        self.dec3 = self.up_conv_block(512, 256)
        self.dec2 = self.up_conv_block(256, 128)
        self.dec1 = self.up_conv_block(128, 64)
        
        # Final output layer
        self.final = nn.Conv2d(64, 1, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        return block
    
    def up_conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            self.conv_block(out_channels, out_channels)
        )
        return block
    
    def forward(self, x):
        # Encoder path
        enc1 = self.enc1(x)
        enc2 = self.enc2(nn.MaxPool2d(kernel_size=2)(enc1))
        enc3 = self.enc3(nn.MaxPool2d(kernel_size=2)(enc2))
        enc4 = self.enc4(nn.MaxPool2d(kernel_size=2)(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(nn.MaxPool2d(kernel_size=2)(enc4))
        
        # Decoder path with skip connections
        dec4 = self.dec4(torch.cat([nn.Upsample(scale_factor=2)(bottleneck), enc4], dim=1))
        dec3 = self.dec3(torch.cat([nn.Upsample(scale_factor=2)(dec4), enc3], dim=1))
        dec2 = self.dec2(torch.cat([nn.Upsample(scale_factor=2)(dec3), enc2], dim=1))
        dec1 = self.dec1(torch.cat([nn.Upsample(scale_factor=2)(dec2), enc1], dim=1))
        
        # Output layer
        output = self.final(dec1)
        return output

# Instantiate and print the model to check the architecture
model = UNet()
print(model)
