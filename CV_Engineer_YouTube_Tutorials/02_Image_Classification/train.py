from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training). YOLOV8 image class model

# data folder has to have a train and val subfolder of images, or model.train will not work.
model.train(data='/Users/tawate/Documents/H2O_Analytics/data/cv_engineer_youtube_data/weather_images',
            epochs=20, imgsz=64)

""" 
1.  training file shows weights, args.yaml, and results.csv file. You can have multiple training runs, and results will not necessairly be the same.
2.  yaml file contains all the paramters of the saved models
3.  csv shows the a row per epoch, with accuracy metrics, training loss, validation loss, etc.
    want training and validation loss to go down as accuracy goes up. Graph these to understand level of improvement for each epoch
4.  weights directory is where the models will be saved. (best.pt and last.pt)
    i. best.pt = best performing model across all epochs
    ii. last.pt = last model trained. (last epoch)
""" 