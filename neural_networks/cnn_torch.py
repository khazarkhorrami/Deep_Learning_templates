#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 12:52:59 2022

template for convolutional neural network in pytorch

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy

#%% example layers

print ("\n..............example layers ......................\n")

n = 24

input_example = torch.rand(n, 3, 32, 32)
output_example = torch.rand(n, 2 )

print(f"Shape of X is : {input_example.shape}")
print(f"Type of X is : {input_example.dtype}")


layer1 = nn.Conv2d(3, 8, 5)
y_1 = layer1(input_example)
y_1 = F.max_pool2d(y_1, (2,2))

print(f"Shape of output is : {y_1.shape}")
print(f"Type of output is : {y_1.dtype}")

layer2 = nn.Conv2d(8, 16, 5)
y_2 = layer2(y_1)
y_2 = F.max_pool2d(y_2, (2,2))

h = y_2.shape[2]
w = y_2.shape[3]
number_of_neurons = 16*h*w

print(f"Shape of output is : {y_2.shape}")
print(f"Type of output is : {y_2.dtype}")

y_2_flat = torch.flatten(y_2, 1)

print(f"Shape of output is : {y_2_flat.shape}")
print(f"Type of output is : {y_2_flat.dtype}")


layer3 = nn.Linear(number_of_neurons, 120)
y_3 = layer3(y_2_flat)

print(f"Shape of output is : {y_3.shape}")
print(f"Type of output is : {y_3.dtype}")


layer4 = nn.Linear(120, 2)
y_4 = layer4(y_3)

print(f"Shape of output is : {y_4.shape}")
print(f"Type of output is : {y_4.dtype}")

#%% example model

print ("\n..............example model ......................\n")


class mycnn(nn.Module):
    
    def __init__(self):
        super(mycnn, self).__init__()
        self.layer1 = nn.Conv2d(3, 8, 5)
        self.layer2 = nn.Conv2d(8, 16, 5)
        self.layer3 = nn.Linear(16*5*5, 120)
        self.layer4 = nn.Linear(120, 2)
    def forward(self,x):
        x = F.max_pool2d (layer1(x), (2,2))
        x = F.max_pool2d (layer2(x), (2,2))
        x = layer3(torch.flatten(x, 1))
        x = layer4(x)
        return x

# defining model
model = mycnn()
params = list(model.parameters())


# creating data
batch_size = 24
model_input = torch.rand(batch_size, 3, 32, 32)
model_target = torch.rand(batch_size, 2 )


# setting training hyper-parameters
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


# training loop
optimizer.zero_grad()

model_output = model(model_input)
loss = criterion(model_output, model_target)

loss.backward()
optimizer.step()
kh
#%% example dataset

print ("\n..............example image dataset ......................\n")

batch_size = 24
x_shape = (batch_size, 3, 64, 64)
y_shape = (batch_size, 1000)

x_data = torch.rand(x_shape)
y_data = torch.rand(y_shape)

# if torch.cuda.is_available():
#     x_data = x_data.to('cuda')
#     y_data = y_data.to('cuda')
    
print(f"Shape of X is : {x_data.shape}")
print(f"Type of X is : {x_data.dtype}")
print(f"Shape of Y is : {y_data.shape}")
print(f"Type of Y is : {y_data.dtype}")


#%% example model

print ("\n............. simple cnn model for images ......................\n")

