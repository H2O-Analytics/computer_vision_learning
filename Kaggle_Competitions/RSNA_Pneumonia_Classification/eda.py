import os
import cv2
import numpy as np
import pandas as pd
import json
import pydicom
import pylab
import glob
from skimage import exposure
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw
DATA_PATH = '/Users/tawate/Documents/H2O_Analytics/data/RSNA_Pneumonia/'

# Import and clean bounding box data set
img_bbox_list = pd.read_csv(DATA_PATH + 'training/BBox_List_2017.csv')
img_bbox_list = img_bbox_list.rename(columns={"Bbox [x":"x", "h]":"h"})
img_bbox_list = img_bbox_list.drop(columns=['Unnamed: 6','Unnamed: 7','Unnamed: 8'])

# Import Data Entry File (2017 - 2020)
data_entry = pd.read_csv(DATA_PATH + 'training/Data_Entry_2017_v2020.csv')
data_entry = data_entry.rename(columns={"OriginalImage[Width" : "Org_Img_Width",
                           "Height]": "Org_Img_Height",
                           "OriginalImagePixelSpacing[x":"Org_Img_Pixel_Spacing_x",
                           "y]": "Org_Img_Pixel_Spacing_y"})

# Join data entry and bbox tables for one large patient table
pneumonia_img_w_bbox = data_entry.merge(img_bbox_list, on = 'Image Index', how='left')

# Read in images
img = cv2.imread('/Users/tawate/Documents/H2O_Analytics/data/RSNA_Pneumonia/training/images01/00000001_000.png')
img_bbox = 

# Drawing boudning boxes on images

# Label Images with findings from Data Table

# Determine full set of labels

