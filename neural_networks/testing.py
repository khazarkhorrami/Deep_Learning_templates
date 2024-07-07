#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 12:04:02 2024

@author: hxkhkh
"""

import torch
import torch.nn as nn

tensor1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
tensor2 = torch.tensor([[7, 8, 9], [10, 11, 12]])
print(tensor1)

x = torch.tensor([[1, 2], [3, 4]])
reshaped = x.view(1, 4)
print(reshaped)
#%%

# Mean Squared Error Loss (MSELoss)

# used primarily for regression tasks

mse_loss = nn.MSELoss()
input_ = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5)
loss = mse_loss(input_, target)
print(loss)

print("Input:")
print(input_)
torch_tensor = input_
numpy_array = torch_tensor.detach().numpy()
print(numpy_array)
#%%

# Cross-Entropy Loss (CrossEntropyLoss)

# used for classification tasks

cross_entropy_loss = nn.CrossEntropyLoss()
input_ = torch.randn(3, 5, requires_grad=True)
target = torch.tensor([1, 0, 4])
loss = cross_entropy_loss(input_, target)
print(loss)

#%%

# Binary Cross-Entropy Loss (BCELoss)
# Used for binary classification tasks

bce_loss = nn.BCELoss()
input_ = torch.sigmoid(torch.randn(3, requires_grad=True))
target = torch.empty(3).random_(2)
loss = bce_loss(input_, target)
print(loss)
print(target)
