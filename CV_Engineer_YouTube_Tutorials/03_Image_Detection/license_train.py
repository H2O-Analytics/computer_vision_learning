from ultralytics import YOLO

model = YOLO("yolov8n.pt") # build new model

# Use the model
resluts = model.train(data = "license_config.yaml", epochs = 1)