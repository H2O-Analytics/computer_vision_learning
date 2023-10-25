import os
import cv2

image_path = '/Users/tawate/Documents/openCV_tutorials/img_2.5.jpg'
img = cv2.imread(image_path)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Code Notes
#   cv2.threshold converts a grayscale image to binary colors of black and white with a threshold of 80 (above goes to black, below white)
#   adaptive thresholding - creates multiple thresholds for the image. each seciton of the image would use an optimal threshold
ret, thresh = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY)
adapt_thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 30)
blur = cv2.blur(img_gray, (10, 10))
ret, thresh_blur = cv2.threshold(blur, 80, 255, cv2.THRESH_BINARY)

cv2.imshow('img', img)
cv2.imshow('img_gray', img_gray)
cv2.imshow('thresh', thresh)
cv2.imshow('thresh_blur', thresh_blur)
cv2.imshow('adapt',adapt_thresh)
cv2.waitKey(0)

