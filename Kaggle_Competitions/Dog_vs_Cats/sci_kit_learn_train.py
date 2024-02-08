import os
import pickle 

from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

DATA_PATH = '/Users/tawate/Documents/H2O_Analytics/data/Kaggle/dogs_v_cats/yolo_format'
categories = ['dog', 'cat']