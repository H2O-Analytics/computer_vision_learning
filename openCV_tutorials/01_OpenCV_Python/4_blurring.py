import os
import cv2
import numpy as np
import pandas as pd

# image has a lot of inherent noise in the background (grainy black dots)
image_path = '/Users/tawate/Documents/openCV_tutorials/cow.jpg'
img = cv2.imread(image_path)

# kernel size or neighborhood size. Larger k size can created "over blurred" images
k_size = 7
k_size2 = 70
# square blur size (7,7)
img_blur = cv2.blur(img, (k_size,k_size))
img_blur2 = cv2.blur(img, (k_size2,k_size2))
img_gauss_blur = cv2.GaussianBlur(img, (k_size,k_size), 5) 
img_median_blur = cv2.medianBlur(img, k_size)

cv2.imshow('img',img)
cv2.imshow('img_blur',img_blur)
cv2.imshow('img_blur2', img_blur)
cv2.imshow('img_gauss_blur',img_gauss_blur)
cv2.imshow('img_median_blur', img_median_blur)
cv2.waitKey(0)