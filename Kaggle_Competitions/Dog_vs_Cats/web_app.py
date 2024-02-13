import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np
from ultralytics import YOLO
import cv2
from cv2 import IMREAD_COLOR,IMREAD_UNCHANGED
import os

from web_app_util import classify, set_background

set_background('/Users/tawate/Documents/H2O_Analytics/data/Kaggle/dogs_v_cats/cat_dog.jpeg')

# set title
st.title('Dog or Cat Classification')

# set header
st.header('Upload a Picture of a Dog or Cat')

# upload file
file = st.file_uploader('',type=['jpeg','jpg', 'png'])

# load classifier
model = YOLO('/Users/tawate/Documents/H2O_Analytics/computer_vision_learning/runs/classify/train5/weights/best.pt')

# Calculates the Laplacian Variance for blur detection
def laplace_variance(image):
    # compute the laplacian of the image and return the focus
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

# load 
if file is not None:
    img_file = Image.open(file)
    img = np.array(img_file)
    fm = laplace_variance(img)
    # Sharpening Kernel Options
    kernel_sharp1=np.array([[0,-1,0],
                            [-1,5,-1],
                            [0,-1,0]])
    kernel_sharp2=np.array([[-1,-1,-1],
                            [-1,9,-1],
                            [-1,-1,-1]])
    # Blur detection threshold
    threshold = 100
    
    # Apply sharpening kernels to images with a laplacian variance < threhold
    if fm < threshold:
        st.write("Blur detection found the image needed sharpening prior to predicition, with a score of ", fm, " and threshold of ", threshold)
        img_org = img
        img_sharp1 = cv2.filter2D(img, -1, kernel_sharp1)
        img_sharp2 = cv2.filter2D(img, -1, kernel_sharp2)
        if laplace_variance(img_sharp1) > laplace_variance(img_sharp2):
            img = img_sharp1
            fm_sharp = laplace_variance(img_sharp1)
        else:
            img = img_sharp2
            fm_sharp = laplace_variance(img_sharp2)
        st.write("Original Image")
        st.image(img_org, use_column_width=True)
        st.write("Sharpened Image with blur detection score of ", fm_sharp)
        st.image(img, use_column_width = True)
    else:
        st.image(img, use_column_width = True)
        
    # generate classification prediction
    result = model(img)
    names_dict = result[0].names
    probs = result[0].probs.data.tolist()
    
    # write classification to gui
    st.write("I am ", np.max(probs) * 100, "% ", "confident this is a ",names_dict[np.argmax(probs)])
  

