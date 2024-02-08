import numpy as np
from ultralytics import YOLO
TEST_DATA = '/Users/tawate/Documents/H2O_Analytics/data/Kaggle/dogs_v_cats/'
# trained yolo model
model = YOLO('/Users/tawate/Documents/H2O_Analytics/computer_vision_learning/runs/classify/train5/weights/best.pt')

result = model(TEST_DATA + 'carlos_face.jpg')

names_dict = result[0].names
probs = result[0].probs.data.tolist()

print(names_dict)
print(probs)
# gets the argument with the max probability value (or result of the prediction)
print(names_dict[np.argmax(probs)])


