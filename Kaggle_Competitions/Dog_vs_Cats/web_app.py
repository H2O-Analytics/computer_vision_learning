import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np
from ultralytics import YOLO

from web_app_util import classify, set_background

set_background('/Users/tawate/Documents/H2O_Analytics/data/cv_engineer_youtube_data/bg5.png')

# set title
st.title('Dog or Cat Classification')

# set header
st.header('Upload a Picture of a Dog or Cat')

# upload file
file = st.file_uploader('',type=['jpeg','jpg', 'png'])

# load classifier
model = YOLO('/Users/tawate/Documents/H2O_Analytics/computer_vision_learning/runs/classify/train5/weights/best.pt')

# load 
