#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Feb  8 11:01:59 2022

template for recurrent neural network in pytorch

"""

import torch
import torch.nn as nn
import numpy

#%% simple model



#%% model class


#%% example dataset

shape_x = (10, 64, 40)
shape_y = (10, 64, 2)

x_data = torch.rand(shape_x)
y_data = torch.rand(shape_y)

if torch.cuda.is_available():
    x_data = x_data.to('cuda')
    y_data = y_data.to('cuda')
    
print(f"Shape of X is :  {x_data.shape}")
print(f"Type of X is : {x_data.dtype}")
print(f"Shape of Y is  : {y_data.shape}")
print(f"Type of Y is  : {y_data.dtype}")




