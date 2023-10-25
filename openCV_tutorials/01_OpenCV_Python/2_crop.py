import os
import cv2

image_path = '/Users/tawate/Documents/openCV_tutorials/img_2.5.jpg'

img = cv2.imread(image_path)
print(img.shape)

# image is set in a numpy array
cropped_img = img[700:1400, 900:2000]
cv2.imshow('cropped_img',cropped_img)
cv2.imshow('img',img)
cv2.waitKey(0)