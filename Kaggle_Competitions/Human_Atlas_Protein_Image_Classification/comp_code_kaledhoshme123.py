import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
DATA_PATH = "/Users/tawate/Library/CloudStorage/OneDrive-SAS/08_CDT_Data/Kaggle Competitions/Human Atlas Protein Location/"

# read csv file
train_csv = pd.read_csv(DATA_PATH + 'train.csv')

# create list of protein ids
protein_id = train_csv['Id']
target = train_csv['Target']

# Create subplot of first 20 protein ids
for i in range(20):
    # create subplot dimensions
    plt.subplot(4,5,i+1)
    
    # 0 converts image to grey scale
    red = cv2.imread(DATA_PATH+ 'train/' + str(protein_id[i]) + '_red.png',0)
    green = cv2.imread(DATA_PATH+ 'train/' + str(protein_id[i]) + '_green.png',0)
    blue = cv2.imread(DATA_PATH+ 'train/' + str(protein_id[i]) + '_blue.png',0)
    yellow = cv2.imread(DATA_PATH+ 'train/' + str(protein_id[i]) + '_yellow.png',0)
    
    # stacking image baseically just assigns red to the first channel, green to the second channel, and blue to the third channel
    image_stack = np.stack((red, green, blue), -1)
    resize = cv2.resize(red,(90,90))
    res = resize / 255
    
    #show individual images
    plt.imshow(image_stack)

plt.show()

# Data PreProcessing
def shape_image(img, target_size):
    """nomalize image array by 255

    Args:
        img (array): image in (255, 255, 3)
        target_size (tupple): 2d array image size (x, y)
    """
    img = cv2.resize(img, target_size)
    img = img/255
    return img

def read_img(paths):
    """read each individual color channel (RGB and Yellow) then stack on top of each to create final protein image

    Args:
        paths (list): comma separated list of images relating to one protein_id
    """
    # Try exception was added for "ugly" images
    try: 
        # Read each individual color channel
        red = cv2.imread(paths[0], 0)
        red = shape_image(red, (90, 90))
        blue = cv2.imread(paths[1], 0)
        blue = shape_image(blue, (90, 90))
        yellow = cv2.imread(paths[2])
        yellow = shape_image(yellow, (90, 90))
        green = cv2.imread(paths[3], 0)
        green = shape_image(green, (90, 90))  
          
        # Return the stacked image
        return np.array([np.stack((red, green, blue), -1), yellow])
    except Exception as e:
        print(str(e))


# read each protein id and create final protein image
images = []
for i in range(len(protein_id)-1):
    arr = read_img([
        DATA_PATH+ 'train/' + str(protein_id[i]) + '_red.png'
        ,DATA_PATH+ 'train/' + str(protein_id[i]) + '_blue.png'
        ,DATA_PATH+ 'train/' + str(protein_id[i]) + '_yellow.png'
        ,DATA_PATH+ 'train/' + str(protein_id[i]) + '_green.png'
        ])
    images.append(arr)
    
images = np.asarray(images)
images.shape
images[0][0].shape

# print full stacked images
plt.figure(figsize = (17, 12))
for i in range(24):
    plt.subplot(4, 6, i + 1)
    plt.imshow(images[i][0])
plt.show()

# print RGB image and yellow image side by side
plt.figure(figsize = (12, 4))
colors = ["rgb", "yellow"]
for i in range(2):
    plt.subplot(1, 4, i + 1)
    plt.imshow(images[1][i])
    plt.title(colors[i])
plt.show()

# target values analysis
target_after = np.zeros((len(train_csv), 28), dtype=28)
target_after.shape
target_after[0]

# convert target read in to list per protein id. list is 28 in length with a 1 where the protein location is identified and 0 where protein location not identified
for index, tar in enumerate(target):
    ids = tar.split()
    for id in ids:
        target_after[index, int(id)] = 1


# Create Neural Network Architecture
DenseNet_Model = tf.keras.applications.DenseNet201(include_top = False
                                                   ,weights = 'imagenet'
                                                   ,input_shape = (90,90,3))

    
    