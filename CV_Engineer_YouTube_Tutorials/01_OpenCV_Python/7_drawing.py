import os
import cv2
import numpy as np

image_path = '/Users/tawate/Documents/openCV_tutorials/img_2.5.jpg'
img = cv2.imread(image_path)

print(img)
#lines, rectanges, circles, etc.

# start point, end point, color, thickness
cv2.line(img, (100, 150), (300, 450), (0, 255, 0), 3)

# top left corner point, bottom right corner point. color. thickness (-1 = filled)
cv2.rectangle(img, (200, 350), (450, 600), (0, 0, 255), -1)

# center of circle, radius, color, thickness
cv2.circle(img, (800, 200), 75, (255, 0, 0), 10)

# text. point. font. color. 
cv2.putText(img, 'Hey you!', (600, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 10)

cv2.imshow('img', img)
cv2.waitKey(0)
