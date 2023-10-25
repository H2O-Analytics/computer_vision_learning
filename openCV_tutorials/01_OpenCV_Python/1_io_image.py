import os
import cv2

image_path = '/Users/tawate/Documents/openCV_tutorials/img_2.5.jpg'

img = cv2.imread(image_path)

cv2.imwrite('/Users/tawate/Documents/openCV_tutorials/img_2.5_out.jpg',img)

cv2.imshow('image',img)
cv2.waitKey(5000)

