import os
import cv2
import numpy as np

image_path = '/Users/tawate/Documents/openCV_tutorials/img_2.5.jpg'
img = cv2.imread(image_path)

# Canny edge detection
img_edge = cv2.Canny(img, 100, 200)
# makes edges thicker. thickness of the lines can be adjusted with the numpy array size
img_edge_d = cv2.dilate(img_edge, np.ones((3, 3), dtype = np.int8))
# makes edges thinner. 
img_edge_e = cv2.erode(img_edge, np.ones((3, 3,), dtype = np.int8))

cv2.imshow('img', img)
cv2.imshow('img_edge', img_edge)
cv2.imshow('dialte', img_edge_d)
cv2.imshow('erode', img_edge_e)
cv2.waitKey(0)
