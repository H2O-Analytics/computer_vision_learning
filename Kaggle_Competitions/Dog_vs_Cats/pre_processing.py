# open cv packege
import cv2
from cv2 import IMREAD_COLOR,IMREAD_UNCHANGED
import os

# useful packeges
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

# statistic packeges
from scipy.ndimage import variance
from skimage import io
from skimage.color import rgb2gray
from skimage.filters import laplace
from skimage.transform import resize
# %matplotlib inline

# import config
DATA_PATH = '/Users/tawate/Documents/H2O_Analytics/data/Kaggle/dogs_v_cats/yolo_format'
img = cv2.imread(DATA_PATH + '/val/cat/cat.0.jpg')
img_canny = cv2.Canny(img, 100, 200)

# cv2.imshow('img', img)
# cv2.waitKey(5000)
# cv2.imshow('canny_edge', img_canny)
# cv2.waitKey(5000)

# Detect Blurred Image: https://www.kaggle.com/code/eladhaziza/perform-blur-detection-with-opencv#
def laplace_variance(image):
    # compute the laplacian of the image and return the focus
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


kernel_sharp1=np.array([[0,-1,0],
                        [-1,5,-1],
                        [0,-1,0]])
kernel_sharp2=np.array([[-1,-1,-1],
                        [-1,9,-1],
                        [-1,-1,-1]])
threshold = 100
for filename in os.listdir(DATA_PATH + '/train/dog'):
    if 'Store' in filename:
        print('Store File')
    else:
        img = cv2.imread(DATA_PATH + '/train/dog/' + filename)
        fm = laplace_variance(img)
        if fm < threshold:
            img_sharp1 = cv2.filter2D(img, -1, kernel_sharp1)
            img_sharp2 = cv2.filter2D(img, -1, kernel_sharp2)
            if laplace_variance(img_sharp1) > laplace_variance(img_sharp2):
                img = img_sharp1
            else:
                img = img_sharp2
        cv2.imwrite('/Users/tawate/Documents/H2O_Analytics/data/Kaggle/dogs_v_cats/new_dogs/' + filename, img)

img_sharp = cv2.filter2D(img, -1, kernel_sharp1)
img_sharp2 = cv2.filter2D(img, -1, kernel_sharp2)
plt.subplot(1,2,1)
plt.title('Original')
plt.imshow(img)
plt.subplot(1,2,2)
plt.title('Sharpened')
plt.imshow(img_sharp2)

laplace_variance(img)
laplace_variance(img_sharp)
laplace_variance(img_sharp2)
cv2.imshow('a',img)
cv2.waitKey(5000)
cv2.imshow('b',img_sharp)
cv2.waitKey(5000)
cv2.imshow('c', img_sharp2)
cv2.waitKey(5000)