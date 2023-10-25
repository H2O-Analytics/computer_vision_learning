import os
import cv2

# read video
video_path = '/Users/tawate/Documents/openCV_tutorials/race_car-TLD.mp4'

video = cv2.VideoCapture(video_path)

ret = True

while ret:
    ret, frame = video.read()
    
    if ret:
        cv2.imshow('frame', frame)
        cv2.waitKey(40) #video is 25 frames/sec or 1 frame every 40 miliseconds. Setting wait 40 shows us one frame at a time.
        
video.release()
cv2.destroyAllWindows