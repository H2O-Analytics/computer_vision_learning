import os
import cv2

image_path = '/Users/tawate/Documents/openCV_tutorials/img_2.5.jpg'

#define image and resized image
img = cv2.imread(image_path)
resize_img = cv2.resize(img,(1408,1408))

# print size. Printed shape is in height, width. But resize function is in width, height!!
print(img.shape)
print(resize_img.shape)

#show images
cv2.imshow('img',img)
cv2.imshow('resize_img',resize_img)
cv2.waitKey(0)