import os
import cv2
import config
from ultralytics import YOLO

model = YOLO("yolov8n.pt") # build new model

# Use the model
resluts = model.train(data = config.PROGRAM_PATH + "/config.yaml", epochs = 1)