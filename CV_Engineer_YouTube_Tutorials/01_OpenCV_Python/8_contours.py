import os
import cv2
import numpy as np

image_path = '/Users/tawate/Documents/openCV_tutorials/img_2.5.jpg'
img = cv2.imread(image_path)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Take everything outside of the bounds not bewteen.
ret, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)

# Find Contours in image. findCountours function. Depending on version of opencv, third output arguement might be needed
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Creates a list of isolated white contours in image. Creates an object detect using image processing
for cnt in contours:
    print(cv2.contourArea(cnt))
    if cv2.contourArea(cnt) > 200:
        # draws the individual contours
        cv2.drawContours(img, cnt, -1, (0,255,0), 1)
        x1, y1, w, h = cv2.boundingRect(cnt) # get the bounding box coordinates of the contour
        cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (0, 255, 0)) # draw the bounding box
        
cv2.imshow('img', img)
cv2.imshow('thresh', thresh)
cv2.waitKey(0)