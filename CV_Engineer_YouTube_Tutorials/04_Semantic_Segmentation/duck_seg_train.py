import os
from ultralytics import YOLO
# Loads tensorboard for real time model training diagnostics
%load_ext tensorboard
%tensorboard --logdir /Users/tawate/Documents/H2O_Analytics/computer_vision_learning/Projects/Brain_Tumor_Object_Detection/runs/detect
MODEL_YAML = '/Users/tawate/Documents/H2O_Analytics/computer_vision_learning/CV_Engineer_YouTube_Tutorials/04_Semantic_Segmentation/config.yaml'
model = YOLO('yolov8n-seg.pt')

model.train(data = MODEL_YAML, epochs = 10)