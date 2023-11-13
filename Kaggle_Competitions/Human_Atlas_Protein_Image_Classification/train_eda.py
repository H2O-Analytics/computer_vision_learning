import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
DATA_PATH = '/Users/tawate/Library/CloudStorage/OneDrive-SAS/08_CDT_Data/Kaggle Competitions/Human Atlas Protein Location/'

# read csv file
train_csv = pd.read_csv(DATA_PATH + 'train.csv')

# create list of protein ids
protein_id = train_csv['Id'].to_list()

for i in range(20):
    # create subplot dimensions
    plt.subplot(4,5,i+1)
    
    # 0 converts image to grey scale
    red = cv2.imread(DATA_PATH+ 'train/' + str(protein_id[i]) + '_red.png',0)
    green = cv2.imread(DATA_PATH+ 'train/' + str(protein_id[i]) + '_green.png',0)
    blue = cv2.imread(DATA_PATH+ 'train/' + str(protein_id[i]) + '_blue.png',0)
    
    # stacking image baseically just assigns red to the first channel, green to the second channel, and blue to the third channel
    image_stack = np.stack((red, green, blue), -1)
    
    #show individual images
    plt.imshow(image_stack)

plt.show()


