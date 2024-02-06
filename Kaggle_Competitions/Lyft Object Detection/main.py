import subprocess
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
import HTML

PROGRAM_PATH = '/Users/tawate/Documents/H2O_Analytics/computer_vision_learning/Kaggle_Competitions/Lyft Object Detection'
subprocess.call('/Users/tawate/Documents/H2O_Analytics/computer_vision_learning/Kaggle_Competitions/Lyft Object Detection/lyft_dataset_classes.py', shell=True)

DATA_PATH = '/Users/tawate/Documents/H2O_Analytics/data/Kaggle/Lyft_Autonomous_Driving_Object_Detection'
lyft_dataset = LyftDataset(data_path=DATA_PATH, json_path=DATA_PATH + '/train_data')

# Animation Kernel
def generate_next_token(scene):
    scene = lyft_dataset.scene[scene]
    sample_token = scene['first_sample_token']
    sample_record = lyft_dataset.get("sample", sample_token)
    
    while sample_record['next']:
        sample_token = sample_record['next']
        sample_record = lyft_dataset.get("sample", sample_token)
        
        yield sample_token

def animate_images(scene, frames, pointsensor_channel='LIDAR_TOP', interval=1):
    cams = [
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK_RIGHT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_FRONT_LEFT',
    ]

    generator = generate_next_token(scene)

    fig, axs = plt.subplots(
        2, len(cams), figsize=(3*len(cams), 6), 
        sharex=True, sharey=True, gridspec_kw = {'wspace': 0, 'hspace': 0.1}
    )
    
    plt.close(fig)

    def animate_fn(i):
        for _ in range(interval):
            sample_token = next(generator)
            
        for c, camera_channel in enumerate(cams):    
            sample_record = lyft_dataset.get("sample", sample_token)

            pointsensor_token = sample_record["data"][pointsensor_channel]
            camera_token = sample_record["data"][camera_channel]
            
            axs[0, c].clear()
            axs[1, c].clear()
            
            lyft_dataset.render_sample_data(camera_token, with_anns=False, ax=axs[0, c])
            lyft_dataset.render_sample_data(camera_token, with_anns=True, ax=axs[1, c])
            
            axs[0, c].set_title("")
            axs[1, c].set_title("")

    anim = animation.FuncAnimation(fig, animate_fn, frames=frames, interval=interval)
    
    return anim

def animate_lidar(scene, frames, pointsensor_channel='LIDAR_TOP', with_anns=True, interval=1):
    generator = generate_next_token(scene)

    fig, axs = plt.subplots(1, 1, figsize=(8, 8))
    plt.close(fig)

    def animate_fn(i):
        for _ in range(interval):
            sample_token = next(generator)
        
        axs.clear()
        sample_record = lyft_dataset.get("sample", sample_token)
        pointsensor_token = sample_record["data"][pointsensor_channel]
        lyft_dataset.render_sample_data(pointsensor_token, with_anns=with_anns, ax=axs)

    anim = animation.FuncAnimation(fig, animate_fn, frames=frames, interval=interval)
    
    return anim


# check out all the scenes within the data
lyft_dataset.list_scenes()
# extract the first scene of lidar data
my_scene = lyft_dataset.scene[0]

# Render Scenes
def render_scene(index):
    my_scene = lyft_dataset.scene[index]
    my_sample_token = my_scene["first_sample_token"]
    lyft_dataset.render_sample(my_sample_token)

render_scene(0)
render_scene(1)

my_sample_token = my_scene["first_sample_token"]
my_sample = lyft_dataset.get('sample', my_sample_token)
lyft_dataset.render_pointcloud_in_image(sample_token = my_sample["token"]
                                        ,dot_size = 1
                                        ,camera_channel = 'CAM_FRONT')

# Print All Annotations
my_sample['data']

# Front Camera
sensor_channel = 'CAM_FRONT'
my_sample_data = lyft_dataset.get('sample_data', my_sample['data'][sensor_channel])
lyft_dataset.render_sample_data(my_sample_data['token'])

# Back Camera
sensor_channel = 'CAM_BACK'
my_sample_data = lyft_dataset.get('sample_data', my_sample['data'][sensor_channel])
lyft_dataset.render_sample_data(my_sample_data['token'])

# Left Camera
sensor_channel = 'CAM_FRONT_LEFT'
my_sample_data = lyft_dataset.get('sample_data', my_sample['data'][sensor_channel])
lyft_dataset.render_sample_data(my_sample_data['token'])

# Right Camera
sensor_channel = 'CAM_FRONT_RIGHT'
my_sample_data = lyft_dataset.get('sample_data', my_sample['data'][sensor_channel])
lyft_dataset.render_sample_data(my_sample_data['token'])

# Back Left Camera
sensor_channel = 'CAM_BACK_LEFT'
my_sample_data = lyft_dataset.get('sample_data', my_sample['data'][sensor_channel])
lyft_dataset.render_sample_data(my_sample_data['token'])

# Back Right Camera
sensor_channel = 'CAM_BACK_RIGHT'
my_sample_data = lyft_dataset.get('sample_data', my_sample['data'][sensor_channel])
lyft_dataset.render_sample_data(my_sample_data['token'])

# Render a specific annotation
my_annotation_token = my_sample['anns'][10]
my_annotation =  my_sample_data.get('sample_annotation', my_annotation_token)
lyft_dataset.render_annotation(my_annotation_token)

# Render a specific instance
my_instance = lyft_dataset.instance[100]
my_instance
instance_token = my_instance['token']
lyft_dataset.render_instance(instance_token)
lyft_dataset.render_annotation(my_instance['last_annotation_token'])

# Top Lidar Image
my_scene = lyft_dataset.scene[0]
my_sample_token = my_scene["first_sample_token"]
my_sample = lyft_dataset.get('sample', my_sample_token)
lyft_dataset.render_sample_data(my_sample['data']['LIDAR_TOP'], nsweeps=5)

# Front Left Lidar Image
my_scene = lyft_dataset.scene[0]
my_sample_token = my_scene["first_sample_token"]
my_sample = lyft_dataset.get('sample', my_sample_token)
lyft_dataset.render_sample_data(my_sample['data']['LIDAR_FRONT_LEFT'], nsweeps=5)

# Front Right Lidar Image
my_scene = lyft_dataset.scene[0]
my_sample_token = my_scene["first_sample_token"]
my_sample = lyft_dataset.get('sample', my_sample_token)
lyft_dataset.render_sample_data(my_sample['data']['LIDAR_FRONT_RIGHT'], nsweeps=5)

# Animate Camera Scenes for 3 seconds
anim = animate_images(scene=3, frames=100, interval=1)
HTML(anim.to_jshtml(fps=8))

anim = animate_images(scene=7, frames=100, interval=1)
HTML(anim.to_jshtml(fps=8))

anim = animate_images(scene=4, frames=100, interval=1)
HTML(anim.to_jshtml(fps=8))

# Animate Lidar Scenes for 3 seconds
anim = animate_lidar(scene=5, frames=100, interval=1)
HTML(anim.to_jshtml(fps=8))

anim = animate_lidar(scene=25, frames=100, interval=1)
HTML(anim.to_jshtml(fps=8))

anim = animate_lidar(scene=10, frames=100, interval=1)
HTML(anim.to_jshtml(fps=8))