"""
Created on Mon Jun 17 10:12:11 2024
@author: Khazar Khorrami

"""
# To install flask do this: conda install anaconda::flask
from flask import Flask, redirect, url_for, request, render_template
# To install keras do this: conda install anaconda::keras
from tensorflow.keras.applications import resnet50



# Define a flask app
app = Flask(__name__)
# Define the model
model = "../../models/ResNet50.h5" # model path (model_resnet.h5)
model = resnet50() # this produces an error ...> fix the env and call REsNet50 instead 

