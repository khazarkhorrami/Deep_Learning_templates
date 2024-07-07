#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 11:20:01 2024

@author: hxkhkh

AUDIO DENOISING AUTOENCODER
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Define the dataset
class AudioDataset(Dataset):
    def __init__(self, noisy_signals, clean_signals):
        self.noisy_signals = noisy_signals
        self.clean_signals = clean_signals

    def __len__(self):
        return len(self.noisy_signals)

    def __getitem__(self, idx):
        return self.noisy_signals[idx], self.clean_signals[idx]

# Define the autoencoder model
class AudioDenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(AudioDenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(16000, 8000),
            nn.ReLU(),
            nn.Linear(8000, 4000),
            nn.ReLU(),
            nn.Linear(4000, 1000),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(1000, 4000),
            nn.ReLU(),
            nn.Linear(4000, 8000),
            nn.ReLU(),
            nn.Linear(8000, 16000),
            nn.Tanh()  # Tanh can be used to keep the output between -1 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Instantiate the model, loss function, and optimizer
model = AudioDenoisingAutoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Prepare the data
# Assuming `noisy_signals` and `clean_signals` are lists or arrays of audio data
# Each audio signal is expected to be a 1D array of length 16000 (e.g., 1 second of audio at 16 kHz)
# You need to replace this with your actual data loading code
noisy_signals = ...
clean_signals = ...

dataset = AudioDataset(noisy_signals, clean_signals)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    for noisy, clean in dataloader:
        noisy = noisy.float()
        clean = clean.float()

        # Forward pass
        outputs = model(noisy)
        loss = criterion(outputs, clean)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training complete.")
