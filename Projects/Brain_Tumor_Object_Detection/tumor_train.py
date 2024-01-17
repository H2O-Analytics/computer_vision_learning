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
           ,epochs = 10
           ,iterations=10
           ,project = save_dir
           ,plots = False
           ,save = False
           ,val = False)

# train the model
resluts = model.train(data = "config.yaml" 
                      ,epochs = 20
                      ,project = save_dir)

# training model from best paramters using the above hyper parameter tuning
results = model.train(  data = "config.yaml"
                        ,epochs = 1 #pick whatever you want
                        ,lr0= 0.00905
                        ,lrf= 0.00917
                        ,momentum= 0.839
                        ,weight_decay= 0.00047
                        ,warmup_epochs= 3.46935
                        ,warmup_momentum= 0.95
                        ,box= 7.29726
                        ,cls= 0.34167
                        ,dfl= 1.5
                        ,hsv_h= 0.015
                        ,hsv_s= 0.70065
                        ,hsv_v= 0.32482
                        ,degrees= 0.0
                        ,translate= 0.1168
                        ,scale= 0.59227
                        ,shear= 0.0
                        ,perspective= 0.0
                        ,flipud= 0.0
                        ,fliplr= 0.5603
                        ,mosaic= 0.97117
                        ,mixup= 0.0
                        ,copy_paste= 0.0
)