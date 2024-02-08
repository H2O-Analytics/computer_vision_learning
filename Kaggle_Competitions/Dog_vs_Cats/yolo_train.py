from ultralytics import YOLO

# Loads tensorboard for real time model training diagnostics
%load_ext tensorboard
%tensorboard --logdir /Users/tawate/Documents/H2O_Analytics/computer_vision_learning/Kaggle_Competitions/Dog_vs_Cats/runs/detect

DATA_PATH = '/Users/tawate/Documents/H2O_Analytics/data/Kaggle/dogs_v_cats/yolo_format'
# load pre trained YOLO image classifier model
model = YOLO('yolov8n-cls.pt')

model.train(data = '/Users/tawate/Documents/H2O_Analytics/data/Kaggle/dogs_v_cats/yolo_format', epochs = 10 )