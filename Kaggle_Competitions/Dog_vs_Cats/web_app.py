import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np
from ultralytics import YOLO

from web_app_util import classify, set_background

set_background('/Users/tawate/Documents/H2O_Analytics/data/Kaggle/dogs_v_cats/bg5.png')

# set title
st.title('Dog or Cat Classification')

# set header
st.header('Upload a Picture of a Dog or Cat')

# upload file
file = st.file_uploader('',type=['jpeg','jpg', 'png'])

# load classifier
model = YOLO('/Users/tawate/Documents/H2O_Analytics/computer_vision_learning/runs/classify/train5/weights/best.pt')

# load 
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width = True)
    # generate classification prediction
    result = model(image)
    names_dict = result[0].names
    probs = result[0].probs.data.tolist()
    
    # write classification to gui
    st.write("I am ", np.max(probs) * 100, "% ", "confident this is a ",names_dict[np.argmax(probs)])
  

