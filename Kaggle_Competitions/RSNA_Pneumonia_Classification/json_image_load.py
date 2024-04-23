# Program is not needed. Data is provided in jpeg format for kaggle competition. Good example of how to parse an image file though.
import os
import cv2
import numpy as np
import pandas as pd
import json
import pydicom
import pylab
import glob
from skimage import exposure
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw
DATA_PATH = '/Users/tawate/Documents/H2O_Analytics/data/RSNA_Pneumonia/'
# Might get at best image quality but not working
def dicom_to_numpy(ds):
    DCM_Img = ds
    rows = DCM_Img.get(0x00280010).value #Get number of rows from tag (0028, 0010)
    cols = DCM_Img.get(0x00280011).value #Get number of cols from tag (0028, 0011)
    
    Instance_Number = int(DCM_Img.get(0x00200013).value) #Get actual slice instance number from tag (0020, 0013)

    Window_Center = int(DCM_Img.get(0x00281050).value) #Get window center from tag (0028, 1050)
    
    Window_Width = int(DCM_Img.get(0x00281051).value) #Get window width from tag (0028, 1051)

    Window_Max = int(Window_Center + Window_Width / 2)
    Window_Min = int(Window_Center - Window_Width / 2)


    if (DCM_Img.get(0x00281052) is None):
        Rescale_Intercept = 0
    else:
        Rescale_Intercept = int(DCM_Img.get(0x00281052).value)

    if (DCM_Img.get(0x00281053) is None):
        Rescale_Slope = 1
    else:
        Rescale_Slope = int(DCM_Img.get(0x00281053).value)

    New_Img = np.zeros((rows, cols), np.uint8)
    Pixels = DCM_Img.pixel_array

    for i in range(0, rows):
        for j in range(0, cols):
            Pix_Val = Pixels[i][j]
            Rescale_Pix_Val = Pix_Val * Rescale_Slope + Rescale_Intercept

            if (Rescale_Pix_Val > Window_Max): #if intensity is greater than max window
                New_Img[i][j] = 255
            elif (Rescale_Pix_Val < Window_Min): #if intensity is less than min window
                New_Img[i][j] = 0
            else:
                New_Img[i][j] = int(((Rescale_Pix_Val - Window_Min) / (Window_Max - Window_Min)) * 255) #Normalize the intensities
                
    return New_Img

file_path = '/Users/tawate/Documents/H2O_Analytics/data/RSNA_Pneumonia/mdai_rsna_project_x9N20BZa_images_2018-07-20-153330/1.2.276.0.7230010.3.1.2.8323329.1472.1517874291.114974/1.2.276.0.7230010.3.1.3.8323329.1472.1517874291.114973/1.2.276.0.7230010.3.1.4.8323329.1472.1517874291.114975.dcm'
file_path_w_annotation = '/Users/tawate/Documents/H2O_Analytics/data/RSNA_Pneumonia/mdai_rsna_project_x9N20BZa_images_2018-07-20-153330/1.2.276.0.7230010.3.1.2.8323329.19020.1517874414.805836/1.2.276.0.7230010.3.1.3.8323329.19020.1517874414.805835/1.2.276.0.7230010.3.1.4.8323329.19020.1517874414.805837'
# id in json annotation file is A_r18n3x
dcm_file = pydicom.read_file(file_path)
print(dcm_file)
# image = dicom_to_numpy(dcm_file)

# Dont need the mapping for getting annotations to image
# Bounding Box json data:
#   x = x position of top left of box
#   y = y position of top left of box
# f = open(DATA_PATH + 'pneumonia-challenge-dataset-mappings_2018.json')
# f_bb = open(DATA_PATH + 'pneumonia-challenge-annotations-adjudicated-kaggle_2018.json')
f = open('/Users/tawate/Documents/H2O_Analytics/computer_vision_learning/Kaggle_Competitions/RSNA_Pneumonia_Classification/annotations.json')
json.load(f)
json.load(f_bb)

# Move out just the annotations from original json file and arrange in annotations.json. Change single quotes to double qoutes. Change None to null. 
# Put full json into []
# annot_dict is a list of dictionairies
with open('/Users/tawate/Documents/H2O_Analytics/computer_vision_learning/Kaggle_Competitions/RSNA_Pneumonia_Classification/annotations.json' ,  'r') as f:
    annot_dict = json.load(f)

# remove images without a series and sop instance
annot_dict_clean = []
annot_dict_removed = []
for row in annot_dict:
    if 'SeriesInstanceUID' in row.keys() and "SOPInstanceUID" in row.keys():
        annot_dict_clean.append(row)
    else: annot_dict_removed.append(row)

for id in annot_dict_clean:
    row = id
    study = id["StudyInstanceUID"]
    series = id["SeriesInstanceUID"]
    sop = id["SOPInstanceUID"]
    dcm_file = pydicom.read_file(DATA_PATH + 'mdai_rsna_project_x9N20BZa_images_2018-07-20-153330/' + study + '/' + series + '/' + sop + '.dcm')
    img_array = dcm_file.pixel_array
    img = Image.fromarray(img_array)
    annotation_lbl = id["annotationNumber"]# check annotation number to know which training label to give it
    if annotation_lbl == None:
        annotation_lbl = 0
    if id["data"] != None:
        # get bbox dimensions
        bbox_x = id["data"]['x']
        bbox_y = id["data"]['y']
        bbox_width = id["data"]['width']
        bbox_height = id["data"]['height']
        draw = ImageDraw.Draw(img)
        # draw bounding box
        draw.rectangle([(bbox_x, bbox_y), (bbox_x+bbox_width, bbox_y+bbox_height)], outline='red',width=5) 
        # output image to directory
        img.save(DATA_PATH + str(annotation_lbl) + '/' + sop + '.jpg')
    else:
        img.save(DATA_PATH + str(annotation_lbl) + '/' + sop + '.jpg')

# Read in dicom file
study = '1.2.276.0.7230010.3.1.2.8323329.2327.1517874295.927196'
series = '1.2.276.0.7230010.3.1.3.8323329.2327.1517874295.927195'
sop = '1.2.276.0.7230010.3.1.4.8323329.2327.1517874295.927197'
dcm_file = pydicom.read_file(DATA_PATH + 'mdai_rsna_project_x9N20BZa_images_2018-07-20-153330/' + study + '/' + series + '/' + sop + '.dcm')
img_array = dcm_file.pixel_array
fig, a = plt.subplots(1,1)
rect = patches.Rectangle((bbox_x, bbox_y),
                         width=bbox_width,
                         height=bbox_width,
                         linewidth = 2,
                         edgecolor = 'r',
                         facecolor = 'none')

img = Image.fromarray(img_array)
draw = ImageDraw.Draw(img)
draw.rectangle([(bbox_x, bbox_y), (bbox_x+bbox_width, bbox_y+bbox_height)], outline='red',width=5) 
img.show()


img_cv = exposure.equalize_adapthist(img_array)
cv2.imshow(sop, img_cv)
cv2.waitKey(5000)
cv2.destroyAllWindows()


