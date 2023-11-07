from ultralytics import YOLO
import numpy as np

model = YOLO('./runs/classify/train14/weights/last.pt')  # load a custom model. (last.py)
# model = YOLO('./runs/classify/train14/weights/bet.pt')  # load a custom model. (best.py)

results = model('./data/weather_dataset/train/sunrise/sunrise1.jpg')  # predict on an image

names_dict = results[0].names

# probability vector with 4 elements. Each element is the probability of image being in categorized to each of the labels (cloudy, rain, shine, sunrise)
probs = results[0].probs.data.tolist() 

print(names_dict)
print(probs)

# gets the argument with the max probability value (or result of the prediction)
print(names_dict[np.argmax(probs)])