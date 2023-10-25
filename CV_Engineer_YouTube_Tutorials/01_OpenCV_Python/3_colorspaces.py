import os
import cv2

image_path = '/Users/tawate/Documents/openCV_tutorials/img_2.5.jpg'
img = cv2.imread(image_path)
cv2.imshow('img',img)
# Notes
# Imread produces an image in BGR colorspace by default
# cvtColor can be used to convert to other color spaces


img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # shows intesity of pixel (magnitude). One channel.
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # switches blue and green channel positions.
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # good for color detection.

cv2.imshow('img_gray',img_gray)
cv2.imshow('img_rgb',img_rgb)
cv2.imshow('img_hsv',img_hsv)
cv2.waitKey(0)
