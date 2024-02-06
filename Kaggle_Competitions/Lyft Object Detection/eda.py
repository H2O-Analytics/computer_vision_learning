import os
import gc
import numpy as np
import pandas as pd

import json
import math
import sys
import time
from datetime import datetime
from typing import Tuple, List

import cv2
import matplotlib.pyplot as plt
import sklearn.metrics
from PIL import Image

from matplotlib.axes import Axes
from matplotlib import animation, rc
import plotly.graph_objs as go
import plotly.tools as tls
from plotly.offline import plot, init_notebook_mode
import plotly.figure_factory as ff

init_notebook_mode(connected=True)

import seaborn as sns
from pyquaternion import Quaternion
from tqdm import tqdm # shows progress bar when looping

from lyft_dataset_sdk.utils.map_mask import MapMask
from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.geometry_utils import view_points, box_in_image, BoxVisibility
from lyft_dataset_sdk.utils.geometry_utils import view_points, transform_matrix
from pathlib import Path

import struct
from abc import ABC, abstractmethod
from functools import reduce
from typing import Tuple, List, Dict
import copy
# import config

DATA_PATH = '/Users/tawate/Documents/H2O_Analytics/data/Kaggle/Lyft_Autonomous_Driving_Object_Detection'

# Load training configuration data. Contains bounding box details for each training image
train = pd.read_csv(DATA_PATH + '/train.csv')
sample_submission = pd.read_csv(DATA_PATH + '/sample_submission.csv')


# Create object columns information
object_columns = ['sample_id','object_id','center_x','center_y','center_z','width','length','height','yaw','class_name']

objects = []
# sample_id = first column in train dataframe
# ps = prediction string column in train dataframe
for sample_id, ps, in tqdm(train.values[:]):
    object_params = ps.split()
    n_objects = len(object_params)
    for i in range(n_objects // 8):
        x, y, z, w, l, h, yaw, c = tuple(object_params[i * 8: (i + 1) * 8])
        objects.append([sample_id, i, x, y, z, w, l, h, yaw, c])
        
# Write objects to a data frame
train_objects = pd.DataFrame(objects, 
                             columns = object_columns)

# Convert numerical featurs from str to float
numerical_cols = ['object_id', 'center_x', 'center_y', 'center_z', 'width', 'length', 'height', 'yaw']
train_objects[numerical_cols] = np.float32(train_objects[numerical_cols].values)
        
""" Visualize Distribution of bounding box coordinates
y_center    = depth
x_center    = left to right in image
z_center    = xz coordinate of the cneter of an objects location. Represents the height of the object above the x-y plane.
yaw         = angle of the volume around the z-axis. Direction the front of the vehicle is pointing at while on ground
width       = bounding box width
length      = bounding box length
heigh       = bounding box heigh
1. Distribution Plot (XY):   
    Shows that the Y cooridnate of the bounding boxes are more right skewed than the x coordinates. 
    This is probably due to the fact that the camera can more evenly see ojects to the left and the right of the camera mount 
    while it has a less evenly distributed visual plane up and down.

2. KDE Kernel Density Plot (XY):
    Shows that there is a negative correlation between depth and side views. Can detect far away within short field of view(x) 
    or close and far field of view, but not both far away and far field of view.
    Also shows that better at far field of view rather than depth. Max x = 3250, max y = 2000
    
3. Distribution Plot (Z): 
    Significantly less spread in the z direction. Most likely due to the fact that objects remain in the plane of the road ahead.
    Camera is attached to the top of the car. Making most measurements become negative.
    
4. Distribution Plot (Yaw):
    Showing large frequencies around -2, -.5, 1.5, and 2.5. Can be classified a bimodal at peaks around -.5 and 2.5. Somewhat balanced overall.
    Showing distinct peaks due to cars being in lanes adjacent to the car with the camer. Car in lane to left will have a positive yaw to the right(clockwise).
    and vice versa for cars in the right lane (counter clock wise).
5. Distribution Plot (Width):
    Approx. normal distribution with a mean of 2
    
6. Distribution Plot (Length):
    Approx. normal distribution with mean of 5. Showing a slightly longer right tail that width
    
7. Distribtion Plot (Height): 
    Approx. normal distribution with mean of 2
"""
# 1: Distribution Plot of XY
fig, ax = plt.subplots(figsize = (10, 10))
sns.distplot(train_objects['center_x'], color='darkorange', ax=ax).set_title('Center_X and Center_Y', fontsize = 16)
sns.distplot(train_objects['center_y'], color='purple', ax=ax).set_title('Center_X and Center_Y', fontsize = 16)
plt.xlabel('Center_X and Center_Y', fontsize=15)
ax.legend('Center_X', 'Center_Y')
plt.show()

# 2. KDE Plot of XY
new_train_objects = train_objects.query('class_name == "car"') # only car bounding boxes
plot = sns.jointplot(x = new_train_objects['center_x'][:1000], # 1000 data points only
                     y = new_train_objects['center_y'][:1000],
                     kind='kde', color='blueviolet')
plot.set_axis_labels('center_x', 'center_y', fontsize = 16)
plt.show()

# 3: Distribution of Z
fig, ax = plt.subplots(figsize = (10,10))
sns.distplot(train_objects['center_z'], color='navy', ax=ax).set_title('Center_Z', fontsize = 16)
plt.xlabel('Center Z', fontsize = 15)
plt.show()

# 4: Distribution of yaw
fig, ax = plt.subplots(figsize = (10,10))
sns.distplot(train_objects['yaw'], color='green', ax=ax).set_title('Yaw', fontsize = 16)
plt.xlabel('Yaw', fontsize = 15)
plt.show()

# 5: Distribution of width
fig, ax = plt.subplots(figsize = (10,10))
sns.distplot(train_objects['width'], color='red', ax=ax).set_title('Width', fontsize = 16)
plt.xlabel('Width', fontsize = 15)
plt.show()

# 6: Distribution of length
fig, ax = plt.subplots(figsize = (10,10))
sns.distplot(train_objects['length'], color='orange', ax=ax).set_title('Length', fontsize = 16)
plt.xlabel('Length', fontsize = 15)
plt.show()

# 7: Distribution of height
fig, ax = plt.subplots(figsize = (10,10))
sns.distplot(train_objects['height'], color='pink', ax=ax).set_title('Height', fontsize = 16)
plt.xlabel('Height', fontsize = 15)
plt.show()


# Frequency of Classes. Vast majority are cars
fig, ax = plt.subplots(figsize = (10,10))
plot = sns.countplot(y = "class_name", data=train_objects)

"""Visualizing Data Columns vs each other
1.  
2.
3.
4.
5.
6.
7.
"""
classes_to_remove = 'class_name != "motorcycle" and class_name != "emergency_vehicle" and class_name != "animal"'


# 1. Center X vs Classes
fig, ax = plt.subplots(figsize=(15, 15))
plot = sns.violinplot(x="class_name", y="center_x",
                      data=train_objects.query(classes_to_remove),
                      palette='YlGnBu',
                      split=True, ax=ax).set_title('center_x (for different objects)', fontsize=16)

plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.xlabel("Class Name", fontsize=15)
plt.ylabel("center_x", fontsize=15)
plt.show(plot)
fig, ax = plt.subplots(figsize=(15, 15))

plot = sns.boxplot(x="class_name", y="center_x",
                   data=train_objects.query(classes_to_remove),
                   palette='YlGnBu', ax=ax).set_title('center_x (for different objects)', fontsize=16)

plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.xlabel("Class Name", fontsize=15)
plt.ylabel("center_x", fontsize=15)
plt.show(plot)

# 2. Center Y vs Classes
fig, ax = plt.subplots(figsize=(15, 15))
plot = sns.violinplot(x="class_name", y="center_y",
                      data=train_objects.query(classes_to_remove),
                      palette='YlGnBu',
                      split=True, ax=ax).set_title('center_y (for different objects)', fontsize=16)

plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.xlabel("Class Name", fontsize=15)
plt.ylabel("center_y", fontsize=15)
plt.show(plot)
fig, ax = plt.subplots(figsize=(15, 15))

plot = sns.boxplot(x="class_name", y="center_y",
                   data=train_objects.query(classes_to_remove),
                   palette='YlGnBu', ax=ax).set_title('center_y (for different objects)', fontsize=16)

plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.xlabel("Class Name", fontsize=15)
plt.ylabel("center_y", fontsize=15)
plt.show(plot)

# 3. Center Z vs Classes
fig, ax = plt.subplots(figsize=(15, 15))
plot = sns.violinplot(x="class_name", y="center_z",
                      data=train_objects.query(classes_to_remove),
                      palette='YlGnBu',
                      split=True, ax=ax).set_title('center_z (for different objects)', fontsize=16)

plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.xlabel("Class Name", fontsize=15)
plt.ylabel("center_z", fontsize=15)
plt.show(plot)
fig, ax = plt.subplots(figsize=(15, 15))

plot = sns.boxplot(x="class_name", y="center_z",
                   data=train_objects.query(classes_to_remove),
                   palette='YlGnBu', ax=ax).set_title('center_z (for different objects)', fontsize=16)

plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.xlabel("Class Name", fontsize=15)
plt.ylabel("center_z", fontsize=15)
plt.show(plot)

# 4. Length vs Classes
fig, ax = plt.subplots(figsize=(15, 15))
plot = sns.violinplot(x="class_name", y="length",
                      data=train_objects.query(classes_to_remove),
                      palette='YlGnBu',
                      split=True, ax=ax).set_title('length (for different objects)', fontsize=16)

plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.xlabel("Class Name", fontsize=15)
plt.ylabel("length", fontsize=15)
plt.show(plot)
fig, ax = plt.subplots(figsize=(15, 15))

plot = sns.boxplot(x="class_name", y="length",
                   data=train_objects.query(classes_to_remove),
                   palette='YlGnBu', ax=ax).set_title('length (for different objects)', fontsize=16)

plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.xlabel("Class Name", fontsize=15)
plt.ylabel("length", fontsize=15)
plt.show(plot)

# 5. Width vs Classes
fig, ax = plt.subplots(figsize=(15, 15))
plot = sns.violinplot(x="class_name", y="width",
                      data=train_objects.query(classes_to_remove),
                      palette='YlGnBu',
                      split=True, ax=ax).set_title('width (for different objects)', fontsize=16)

plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.xlabel("Class Name", fontsize=15)
plt.ylabel("width", fontsize=15)
plt.show(plot)
fig, ax = plt.subplots(figsize=(15, 15))

plot = sns.boxplot(x="class_name", y="width",
                   data=train_objects.query(classes_to_remove),
                   palette='YlGnBu', ax=ax).set_title('width (for different objects)', fontsize=16)

plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.xlabel("Class Name", fontsize=15)
plt.ylabel("width", fontsize=15)
plt.show(plot)

# 6. Height vs Classes
fig, ax = plt.subplots(figsize=(15, 15))
plot = sns.violinplot(x="class_name", y="height",
                      data=train_objects.query(classes_to_remove),
                      palette='YlGnBu',
                      split=True, ax=ax).set_title('height (for different objects)', fontsize=16)

plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.xlabel("Class Name", fontsize=15)
plt.ylabel("height", fontsize=15)
plt.show(plot)
fig, ax = plt.subplots(figsize=(15, 15))

plot = sns.boxplot(x="class_name", y="height",
                   data=train_objects.query(classes_to_remove),
                   palette='YlGnBu', ax=ax).set_title('height (for different objects)', fontsize=16)

plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.xlabel("Class Name", fontsize=15)
plt.ylabel("height", fontsize=15)
plt.show(plot)

# 7. Yaw vs Classes
fig, ax = plt.subplots(figsize=(15, 15))
plot = sns.violinplot(x="class_name", y="yaw",
                      data=train_objects.query(classes_to_remove),
                      palette='YlGnBu',
                      split=True, ax=ax).set_title('yaw (for different objects)', fontsize=16)

plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.xlabel("Class Name", fontsize=15)
plt.ylabel("yaw", fontsize=15)
plt.show(plot)
fig, ax = plt.subplots(figsize=(15, 15))

plot = sns.boxplot(x="class_name", y="yaw",
                   data=train_objects.query(classes_to_remove),
                   palette='YlGnBu', ax=ax).set_title('yaw (for different objects)', fontsize=16)

plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.xlabel("Class Name", fontsize=15)
plt.ylabel("yaw", fontsize=15)
plt.show(plot)

                                        

