import os
import cv2
import config
from ultralytics import YOLO
import numpy as np
import pandas as pd
import pandasql as ps
import pathlib
import matplotlib.pyplot as plt

# Intialize results dictionary
results = {}

# Load Yolov8 Model for detecting vehicles
coco_model = YOLO('yolov8n.pt')

# Load Tumor Detection pre trained YOLO model
tumor_detector = YOLO('/Users/tawate/Documents/H2O_Analytics/computer_vision_learning/Projects/Brain_Tumor_Object_Detection/runs/detect/train2/weights/best.pt')

# Detect Tumor
def detect_objects(image):
    """Finds object within an image using a pre-defined YOLO model

    Args:
        image (img file): file containing an image
    """
    detections = []
    results = tumor_detector(image)[0]
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        detections.append([int(x1)
                        ,int(y1)
                        ,int(x2)
                        ,int(y2)
                        ,round(score, 3)
                        ,results.names[int(class_id)]])
    return detections

images_path = config.DATA_PATH + '/object_detection/brain_tumor/valid/images/'
filenames = os.listdir(images_path)
bbox_directory = []
for filename in filenames:
    for detection in detect_objects(images_path + filename):
        detection.append(filename)
        bbox_directory.append(detection)


bbox_df = pd.DataFrame(bbox_directory, columns = ['x1', 'y1', 'x2', 'y2', 'class_pred', 'class_id', 'id']).sort_values('id')
# Calculate rolling count by id
bbox_df['rolling_id_count'] = bbox_df.groupby('id').cumcount() + 1
max_id_count = bbox_df.groupby('id')['rolling_id_count'].max()
# Get max number of bounding boxes per id
bbox_df = bbox_df.merge(max_id_count.rename('id_count'), left_on='id', right_index=True)
for i in range(len(bbox_df)):
    if bbox_df['id_count'].iloc[i] == 1:
        img_array = cv2.imread(images_path + bbox_df['id'].iloc[i])
        left_corner = (bbox_df['x1'].iloc[i] , bbox_df['y1'].iloc[i])
        right_corner = (bbox_df['x2'].iloc[i] , bbox_df['y2'].iloc[i])
        bbox_img = cv2.rectangle(img_array, left_corner, right_corner, (255,255,255),2)
        cv2.imwrite(config.DATA_PATH + 'object_detection/brain_tumor/valid/images_bbox/' + bbox_df['id'].iloc[i] , bbox_img)
    elif bbox_df['id_count'].iloc[i] > 1:
        img_array = cv2.imread(images_path + bbox_df['id'].iloc[i])
        left_corner = (bbox_df['x1'].iloc[i] , bbox_df['y1'].iloc[i])
        right_corner = (bbox_df['x2'].iloc[i] , bbox_df['y2'].iloc[i])
        if bbox_df['rolling_id_count'].iloc[i] == 1:
            bbox_img = cv2.rectangle(img_array, left_corner, right_corner, (255,255,255),2)
        elif bbox_df['rolling_id_count'].iloc[i] > 1:
            bbox_img = cv2.rectangle(bbox_img, left_corner, right_corner, (255,255,255),2)
        if bbox_df['rolling_id_count'].iloc[i] == bbox_df['id_count'].iloc[i]:
            cv2.imwrite(config.DATA_PATH + 'object_detection/brain_tumor/valid/images_bbox/' + bbox_df['id'].iloc[i] , bbox_img)
        
