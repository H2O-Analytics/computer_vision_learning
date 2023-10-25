import cv2
from PIL import Image
from util import get_limits

yellow = [0, 255, 255] # input color to detect in BGR colorspace
cap = cv2.VideoCapture(0) # initialize Webcam

while True:
    ret, frame = cap.read()
    
    # convert to HSV color space. Using the hue chanel understand color of image. And determine if it is within the yellow interval.
    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 

    # get the upper and lower limits of hue for the color you to detect
    lowerLimit, upperLimit = get_limits(color=yellow)

    # location of all the pixels where color = input to get_limits
    mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)

    # change format of mask from array to image memory
    mask_ = Image.fromarray(mask)

    # fina the bounding box location of the mask
    bbox = mask_.getbbox()

    # print poitns of bounding box
    print(bbox)
    
    # draw bounding box in frame where object in frame = color
    if bbox is not None:
        x1, y1, x2, y2 = bbox

        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()

    
