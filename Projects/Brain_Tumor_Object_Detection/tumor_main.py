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

# Convert bounding box list to dataframe
bbox_df = pd.DataFrame(bbox_directory, columns = ['x1', 'y1', 'x2', 'y2', 'class_pred', 'class_id', 'id']).sort_values('id')

# Calculate rolling count and max count by id
bbox_df['rolling_id_count'] = bbox_df.groupby('id').cumcount() + 1
max_id_count = bbox_df.groupby('id')['rolling_id_count'].max()

# Combine max id count with bbox dataframe.
bbox_df = bbox_df.merge(max_id_count.rename('id_count'), left_on='id', right_index=True)

# Calculate the IoU of two bounding boxes
def bbox_iou(boxA, boxB):
    # https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    # ^^ corrected.
    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = xB - xA + 1
    interH = yB - yA + 1
    # Correction: reject non-overlapping boxes
    if interW <=0 or interH <=0 :
        return -1.0

    interArea = interW * interH
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# Subset to images with multiple bounding boxes
multi_bbox_df = bbox_df[bbox_df['id_count'] > 1]
multi_bbox_df['unique_idx'] = -1
for i in range(len(multi_bbox_df)):
    multi_bbox_df['unique_idx'].iloc[i] = i

# Find bounding boxes that overlap more than 75%. 
# Take the bounding box associated with the larged class prediction per image if meets overlap condition
index_to_drop = []
IOU_THRESHOLD = .75
for i in range(len(multi_bbox_df)):
    if multi_bbox_df['rolling_id_count'].iloc[i] == 1:
        print('first bbox on image')
    elif multi_bbox_df['rolling_id_count'].iloc[i] == 2:
        bbox1 = multi_bbox_df.iloc[i-1][0:4].tolist() # x1, y1, x2, y2 (hard coded to index numbers - bad)
        bbox2 = multi_bbox_df.iloc[i][0:4].tolist() # x1, y1, x2, y2 (hard coded to index numbers - bad)
        iou = bbox_iou( boxA=bbox1
                        ,boxB=bbox2)
        if iou > IOU_THRESHOLD:
            if multi_bbox_df.iloc[i]['class_pred'] >= multi_bbox_df.iloc[i-1]['class_pred']:
                index_to_drop.append(multi_bbox_df['unique_idx'].iloc[i-1])
            else:
                index_to_drop.append(multi_bbox_df['unique_idx'].iloc[i])

# Remove bounding boxe data
multi_bbox_df2 = multi_bbox_df[~multi_bbox_df['unique_idx'].isin(index_to_drop)]

# Merge single bbox images with multi bbox image data 
single_bbox_df = bbox_df[bbox_df['id_count'] == 1]
single_bbox_df2 = single_bbox_df.drop(['rolling_id_count','id_count'], axis=1)
multi_bbox_df3 = multi_bbox_df2.drop(['rolling_id_count','id_count','unique_idx'], axis=1)
bbox_df_fin = pd.concat([multi_bbox_df3, single_bbox_df2], ignore_index=True, sort=False)
bbox_df_fin = bbox_df_fin.sort_values('id')

# Calculate rolling count and max count by id (again)
bbox_df_fin['rolling_id_count'] = bbox_df_fin.groupby('id').cumcount() + 1
max_id_count = bbox_df_fin.groupby('id')['rolling_id_count'].max()
bbox_df_fin = bbox_df_fin.merge(max_id_count.rename('id_count'), left_on='id', right_index=True)

# Draw bounding box(s) on each image and save images to data folder
for i in range(len(bbox_df_fin)):
    img_array = cv2.imread(images_path + bbox_df_fin['id'].iloc[i])
    top_left = (bbox_df_fin['x1'].iloc[i] , bbox_df['y1'].iloc[i])
    bottom_right = (bbox_df_fin['x2'].iloc[i] , bbox_df_fin['y2'].iloc[i])
    top_right = (bbox_df_fin['x2'].iloc[i] , bbox_df_fin['y1'].iloc[i])
    # Assign bbox color depending on classification
    if bbox_df_fin['class_id'].iloc[i] == 'positive': bbox_color = (0,0,255)
    elif bbox_df_fin['class_id'].iloc[i] == 'negative': bbox_color = (0,255,0)
    # One bound box in image
    if bbox_df_fin['id_count'].iloc[i] == 1:
        bbox_img = cv2.rectangle(img_array, top_left, bottom_right, bbox_color,2)
        bbox_img = cv2.putText(bbox_img,str(bbox_df_fin['class_pred'].iloc[i]),top_right,cv2.FONT_HERSHEY_SIMPLEX,.5,bbox_color,2)
        cv2.imwrite(config.DATA_PATH + 'object_detection/brain_tumor/valid/images_bbox/' + bbox_df_fin['id'].iloc[i] , bbox_img)
    # Multiple bounding boxes in image
    elif bbox_df_fin['id_count'].iloc[i] > 1:
        # Draw bboxes
        if bbox_df_fin['rolling_id_count'].iloc[i] == 1:
            bbox_img = cv2.rectangle(img_array, top_left, bottom_right, bbox_color,2)
            bbox_img = cv2.putText(bbox_img,str(bbox_df_fin['class_pred'].iloc[i]),top_right,cv2.FONT_HERSHEY_SIMPLEX,.5,bbox_color,2)
        elif bbox_df_fin['rolling_id_count'].iloc[i] > 1:
            bbox_img = cv2.rectangle(bbox_img, top_left, bottom_right, bbox_color,2)
            bbox_img = cv2.putText(bbox_img,str(bbox_df_fin['class_pred'].iloc[i]),top_right,cv2.FONT_HERSHEY_SIMPLEX,.5,bbox_color,2)
        if bbox_df_fin['rolling_id_count'].iloc[i] == bbox_df_fin['id_count'].iloc[i]:
            cv2.imwrite(config.DATA_PATH + 'object_detection/brain_tumor/valid/images_bbox/' + bbox_df_fin['id'].iloc[i] , bbox_img)
        
