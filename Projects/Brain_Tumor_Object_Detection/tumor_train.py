import os
import cv2
import config
from ultralytics import YOLO
# Loads tensorboard for real time model training diagnostics
%load_ext tensorboard
%tensorboard --logdir /Users/tawate/Documents/H2O_Analytics/computer_vision_learning/Projects/Brain_Tumor_Object_Detection/runs/detect

save_dir = '/Users/tawate/Documents/H2O_Analytics/computer_vision_learning/Projects/Brain_Tumor_Object_Detection/runs/detect'
model = YOLO("yolov8n.pt") # build new model

# Perform Hyperparameter Tuning
model.tune(data = "config.yaml"
           ,epochs = 20
           ,iterations=300
           ,project = save_dir
           ,plots = False
           ,save = False
           ,val = False)

# train the model
resluts = model.train(data = "config.yaml" 
                      ,epochs = 20
                      ,project = save_dir)