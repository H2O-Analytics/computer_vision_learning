import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import config
DATA_PATH = '/Users/tawate/Documents/H2O_Analytics/data/Kaggle/dogs_v_cats'
img = cv2.imread(DATA_PATH + '/train/cat.0.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_hsv_full = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
# Canny edge detection
img_edge = cv2.Canny(img, 100, 200)
# makes edges thicker. thickness of the lines can be adjusted with the numpy array size
img_edge_d = cv2.dilate(img_edge, np.ones((3, 3), dtype = np.int8))
# makes edges thinner. 
img_edge_e = cv2.erode(img_edge, np.ones((3, 3,), dtype = np.int8))
# cv2.imshow('img', img)
# cv2.waitKey(5000)
# cv2.imshow('gray', img_gray)
# cv2.waitKey(5000)
# cv2.imshow('rgb', img_rgb)
# cv2.waitKey(5000)
# cv2.imshow('hsv', img_hsv)
# cv2.waitKey(5000)
# cv2.imshow('hsv_full', img_hsv_full)
# cv2.waitKey(5000)

# image blurring
k_size = 7
k_size2 = 12
img_blur = cv2.blur(img, (k_size, k_size))
img_blur2 = cv2.blur(img, (k_size2, k_size2))
img_gauss_blur = cv2.GaussianBlur(img, (k_size, k_size), 5)
img_median_blur = cv2.medianBlur(img, k_size)
cv2.imshow('img', img)
cv2.waitKey(5000)
cv2.imshow('blur', img_blur)
cv2.waitKey(5000)
cv2.imshow('blur2', img_blur2)
cv2.waitKey(5000)
cv2.imshow('gauss', img_gauss_blur)
cv2.waitKey(5000)
cv2.imshow('median', img_median_blur)
cv2.waitKey(5000)