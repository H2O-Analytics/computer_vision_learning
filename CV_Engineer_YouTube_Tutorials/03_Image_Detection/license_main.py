# This specific code applies directly to the uk license plate system. But the code can be adapted to other license plate systems
# UK License Plate: Area Code (2 letter) Age Identifier (2 alpha numeric) space Random Letters

# Import Packages
from ultralytics import YOLO
from roboflow import Roboflow
import credentials
import config
import cv2
from sort.sort import *
from license_util import *

# Intialize results dictionary
results = {}

# Load Yolov8 Model for detecting vehicles
coco_model = YOLO('yolov8n.pt')

# Load License Plate Recognition pre trained YOLO model
license_plate_detector = YOLO('/Users/tawate/Documents/H2O_Analytics/computer_vision_learning/CV_Engineer_YouTube_Tutorials/03_Image_Detection/license.pt')

# Load Video
cap = cv2.VideoCapture(config.DATA_PATH + '/object_detection/sample.mp4')

# Create object tracker
mot_tracker = Sort()

# Initialize Vehicle Class Id Indexs from Yolov8 Model
vehicles = [2, 3, 5, 7]

# Read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        # Detect vehicles
        detections = coco_model(frame)[0]
        detections_ = [] # list of detections that are actually vehicles
        for detection in detections.boxes.data.tolist():
            # detection object is the following: x1, y1, x2, y2(bounding box, score (confidence), class_id
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # Track Vehicles using Sort.py
        track_ids = mot_tracker.update(np.asarray(detections_)) # adds the car id column to track each car throughout each frame

        # Detect License Plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # Assign license plate to vehicle. Returns -1 if license bbox not within car bbox
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:

                # Crop license plate: input license plate coordinates only to crop out everything but the license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                # Process license plate on cropped out license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY) # grayed out license plate
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV) # all pixels lower than 64 got to 255, all pixels higher than 64 go to 0 (inverse thresh)
                # Temp code to show cropped images for OCR
                # cv2.imshow('org_crop', license_plate_crop)
                # cv2.imshow('crop_gray', license_plate_crop_gray)
                # cv2.imshow('thresh', license_plate_crop_thresh)

                # Read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
                
                # Save all detection info to results dictionary
                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                  'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_text_score}}

# Write results dictionary to csv
write_csv(results = results
          ,output_path = config.OUTPUT_PATH + '/license_plate_results.csv')