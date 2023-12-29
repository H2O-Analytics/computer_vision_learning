from ultralytics import YOLO
YAML_PATH = "/Users/tawate/Documents/H2O_Analytics/computer_vision_learning/CV_Engineer_YouTube_Tutorials/03_Image_Detection/"
# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Use the model
results = model.train(data=YAML_PATH + "config.yaml", epochs=1)  # train the model

"""
Notes:
    1. Having issues with data location and formating for yolov8 need to do this in my weekend kaggle problem
    2. Output of YOLO model with put label on the bounding box if found. 
    3. Need to increase epoch to improve performance
    4. can be ran in command line using yolo detect train data = config.yaml model = "yolov8n.yaml" epochs = 1
    5. confusion matrix creates an alpaca and background classes to show observation of class and lack of observation (0,1)
    6. ensure training loss is decreasing with epochs. loss and mAP are usually inverse related. 
    You want to see the training loss flatten out at the bottom to ensure you have captured all your performance
"""