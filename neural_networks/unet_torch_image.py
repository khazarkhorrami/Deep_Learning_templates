#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 18:16:59 2024

@author: hxkhkh

Image segmentation

Image + mask (GT)
"""
# !pip install segmentation-models-pytorch
# !pip install -U git+https://github.com/albumentations-team/albumentations
# !pip install --upgrade opencv-contrib-python

# download data
# !git clone https://github.com/parth1620/Human-Segmentation-Dataset-master.git

import torch 
import cv2

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split
from tqdm import tqdm

import helper

# Setup Configurations

CSV_FILE = "/content/Human-Segmentation-Dataset-master/train.csv"
DATA_DIR = "/content/"
DEVICE = "cuda"

EPOCHS = 25 
LR = 0.003
IMAGE_SIZE = 320
BATCH_SIZE = 16

ENCODER = "timm-efficientnet-b0"
ENCODER_WEIGHTS = "imagenet"
ACTIVATION = "sigmoid"

# read data

df = pd.read_csv(CSV_FILE)
df.head()

row = df.iloc [4]
image_path = row.images
mask_path = row.masks

image =cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255.0

# plot data
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
        
ax1.set_title('IMAGE')
ax1.imshow(image)

ax2.set_title('GROUND TRUTH')
ax2.imshow(mask,cmap = 'gray')

# split data 

train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)


# Data Augmentation
# https://www.google.com/url?q=https%3A%2F%2Falbumentations.ai%2Fdocs%2F

import albumentations as A

def get_train_augments():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5)
    ])

def get_test_augments():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE)
    ])


######################################################################### Dataset
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
  def __init__(self, df, augmentations):
    self.df = df
    self.augmentations = augmentations
  
  def __len__(self):
    return len(self.df)

  def __getitem__(self, index):
    row = self.df.iloc[index]
    image_path = row.images
    mask_path = row.masks

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) # (h, w, c)
    mask = np.expand_dims(mask, axis = -1)

    if self.augmentations:
      data = self.augmentations(image = image, mask = mask)
      image = data['image']
      mask = data['mask']
    
    image = np.transpose(image, (2,0,1)).astype(np.float32)
    mask = np.transpose(mask, (2,0,1)).astype(np.float32)
    
    image = torch.tensor(image) / 255.0
    mask = torch.round(torch.tensor(mask) / 255.0)
    
    return image, mask
######################################################################### Data Loader
from torch.utils.data import DataLoader


trainloader = DataLoader(trainset, batch_size = BATCH_SIZE, shuffle = True)
validloader = DataLoader(validset, batch_size = BATCH_SIZE, shuffle = False)

print(f"total number of batches in trainloader : {len(trainloader)}")
print(f"total number of batches in validloader: {len(validloader)}")

for image, mask in trainloader: 
  break
print(f"image shape : {image.shape}")
print(f"mask shape : {mask.shape}")

######################################################################### Model
from torch import nn
#!pip install segmentation_models_pytorch
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss


ENCODER = 'resnet34'  # Example encoder
ENCODER_WEIGHTS = 'imagenet'  # Example weights
ACTIVATION = None

class SegmentationModel(nn.Module):
  def __init__(self):
    super(SegmentationModel, self).__init__()
    self.arc = smp.Unet(
        encoder_name = ENCODER,
        encoder_weights = ENCODER_WEIGHTS,
        in_channels = 3,
        classes = 1,
        activation = ACTIVATION)
    def forward(self, images, masks=None):
      logits = self.arc(images)
      if masks != None:
        loss1 = DiceLoss(mode='binary')(logits, masks)
        loss2 = BCEWithLogitsLoss()(logits, masks)
        loss = loss1 + loss2
        return logits, loss
     
      return logits

model = SegmentationModel()
DEVICE = 'cpu'
model.to(DEVICE)

######################################################################### Train

def train_fn(model, trainloader, optimizer):
  model.train()
  total_loss = 0.0
  for images, masks in tqdm(trainloader):
    images = images.to(DEVICE)
    masks = masks.to(DEVICE)
    optimizer.zero_grad()
    logits, loss = model(images, masks)
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
  return total_loss / len(trainloader)


optimizer = torch.optim.Adam(model.parameters(), lr = LR) 

best_valid_loss = np.Inf
for epoch in range(EPOCHS):
  train_loss = train_fn(model, trainloader, optimizer)
  valid_loss = eval_fn(model, validloader)
  if valid_loss < best_valid_loss:
    best_valid_loss = valid_loss
    torch.save(model.state_dict(), 'best_model.pth')
    print("SAVED-MODEL")

print(f"Epoch : {epoch+1} Train_loss : {train_loss} Valid_loss : {valid_loss}")

######################################################################### Inference


def eval_fn(model, validloader):
  model.eval()
  total_loss = 0.0
  with torch.no_grad():
    for images, masks in tqdm(trainloader):
      images = images.to(DEVICE)
      masks = masks.to(DEVICE)
    
      logits, loss = model(images, masks)
      total_loss += loss.item()
  return total_loss / len(trainloader)

best_valid_loss = np.Inf
for epoch in range(EPOCHS):
  train_loss = train_fn(model, trainloader, optimizer)
  valid_loss = eval_fn(model, validloader)
  if valid_loss < best_valid_loss:
    best_valid_loss = valid_loss
    torch.save(model.state_dict(), 'best_model.pth')
    print("SAVED-MODEL")

print(f"Epoch : {epoch+1} Train_loss : {train_loss} Valid_loss : {valid_loss}")

idx = 20
model.load_state_dict(torch.load('/content/best_model.pth'))
image, mask = validset[idx]
logits_mask = model(image.to(DEVICE).unsqueeze(0))
pred_mask = torch.sigmoid(logits_mask)
pred_mask = (pred_mask > 0.5).float()


helper.show_image(image, mask, pred_mask.detach().cpu().squeeze(0))

