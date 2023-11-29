import os
import cv2
import numpy as np
import pandas as pd
import pathlib as p
import matplotlib.pyplot as plt
import tensorflow as tf
DATA_PATH = "/Users/tawate/Documents/H2O_Analytics/data/Kaggle/Human Atlas Protein Location/"
dataset_folder = DATA_PATH + "train"
datasetObject = p.Path(dataset_folder)
dataset_images = list(datasetObject.glob("*.*"))
len(dataset_images)

# read csv file
csv_dataset = pd.read_csv(DATA_PATH + 'train.csv')
csv_dataset.head()
len(csv_dataset)

# target variable format
target = csv_dataset['Target']
target.head()

# create list of protein Ids
IDs = csv_dataset['Id']
plt.figure(figsize=(17, 12))

IDs = csv_dataset['Id']
plt.figure(figsize = (17, 12))
# Create subplot of first 20 protein ids
IDs = csv_dataset['Id']
plt.figure(figsize = (17, 12))
for i in range(20):
    # create subplot dimensions
    plt.subplot(4, 5, i + 1)
    
    # 0 converts image to grey scale
    red= cv2.imread(DATA_PATH + "train/{}_red.png".format(str(IDs[i])), 0)
    green = cv2.imread(DATA_PATH + "train/{}_green.png".format(str(IDs[i])), 0)
    blue = cv2.imread(DATA_PATH + "train/{}_blue.png".format(str(IDs[i])), 0)

    # stacking image baseically just assigns red to the first channel, green to the second channel, and blue to the third channel
    image = np.stack((red, green, blue), -1)
    plt.imshow(image)
plt.show()

# Data PreProcessing
def shape_image(img, target_size):
    """nomalize image array by 255

    Args:
        img (array): image in (255, 255, 3)
        target_size (tupple): 2d array image size (x, y)
    """
    img = cv2.resize(img, target_size)
    img = img/255
    return img

def read_img(paths):
    """read each individual color channel (RGB and Yellow) then stack on top of each to create final protein image

    Args:
        paths (list): comma separated list of images relating to one protein_id
    """
    # Try exception was added for "ugly" images
    try: 
        # Read each individual color channel
        red = cv2.imread(paths[0], 0)
        red = shape_image(red, (90, 90))
        blue = cv2.imread(paths[1], 0)
        blue = shape_image(blue, (90, 90))
        yellow = cv2.imread(paths[2])
        yellow = shape_image(yellow, (90, 90))
        green = cv2.imread(paths[3], 0)
        green = shape_image(green, (90, 90))  
          
        # Return the stacked image
        return np.array([np.stack((red, green, blue), -1), yellow])
    except Exception as e:
        print(str(e))

# read each protein id and create final protein image
images = []
list_images_csv_dataset = csv_dataset['Id']
for img in list_images_csv_dataset:
    arr = read_img([
        DATA_PATH + "train/{}_red.png".format(str(img))
        ,DATA_PATH + "train/{}_green.png".format(str(img))
        ,DATA_PATH + "train/{}_blue.png".format(str(img))
        ,DATA_PATH + "train/{}_yellow.png".format(str(img))
    ])
    images.append(arr)

images = np.asarray(images)
# shape of the final image array
images.shape
# shape of an individual RGB image with 3 channels
images[0][0].shape

# print full stacked images
plt.figure(figsize = (17, 12))
for i in range(24):
    plt.subplot(4, 6, i + 1)
    plt.imshow(images[i][0])
plt.show()

# print RGB image and yellow image side by side
plt.figure(figsize = (12, 4))
colors = ["rgb", "yellow"]
for i in range(2):
    plt.subplot(1, 4, i + 1)
    plt.imshow(images[1][i])
    plt.title(colors[i])
plt.show()

# target values analysis
target_after = np.zeros((len(csv_dataset), 28), dtype=int)
target_after.shape
target_after[0]

# convert target read in to list per protein id. list is 28 in length with a 1 where the protein location is identified and 0 where protein location not identified
for index, tar in enumerate(target):
    ids = tar.split()
    for id in ids:
        target_after[index, int(id)] = 1
target_after[0]



# Create Neural Network Architecture: url to densenet201 tensorflow documentation:
# https://www.tensorflow.org/api_docs/python/tf/keras/applications/densenet/DenseNet201
DenseNet_Model = tf.keras.applications.DenseNet201(include_top = False
                                                   ,weights = 'imagenet' # pre-trained on image net model
                                                   ,input_shape = (90,90,3))


for layer in DenseNet_Model.layers:
    layer.trainable = True

# url to keras model layers: https://www.tensorflow.org/api_docs/python/tf/keras/layers
# Sequential groups a linear stack of layers on top of one another
# Dropout helps with overfitting 
# GlobalAveragePooling1D is used for temporal data
model = tf.keras.models.Sequential([
    tf.keras.layers.TimeDistributed(DenseNet_Model,
                                    input_shape = (2, 90, 90, 3))
    ,tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(.5))
    ,tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D())
    ,tf.keras.layers.GlobalAveragePooling1D(name = "GlobalAveragePooling1D")
    ,tf.keras.layers.BatchNormalization(name = "BatchNormalization")
    ,tf.keras.layers.Dropout(.5)
    ,tf.keras.layers.Dense(1024, activation = "relu")
    ,tf.keras.layers.Dropout(.5)
    ,tf.keras.layers.Dense(28, activation = "sigmoid")
])

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = .00001)
              ,loss = "binary_crossentropy"
              ,metrics = ["binary_accuracy"])
model.summary()

# Show the NN Architecture Steps
import pydot
import graphviz
tf.keras.utils.plot_model(model, show_shapes = True)


# Run model for # of epochs
history = model.fit(
                    images, 
                    target_after, 
                    epochs = 60, 
                    batch_size = 32,
                    validation_split = 0.1,
                    callbacks = [tf.keras.callbacks.ReduceLROnPlateau(  monitor='val_loss', 
                                                                        factor=0.1, 
                                                                        mode = 'min',
                                                                        patience= 1),
              tf.keras.callbacks.EarlyStopping(patience = 7, 
                                               monitor = 'val_loss', 
                                               mode = 'min', 
                                               restore_best_weights=True)]
)
  
    
# Show Model Training Results
figures = ['loss', 'binary_accuracy']
titles = ["loss vs (validation loss)", "accuracy vs (validation accuracy)"] 

plt.figure(figsize = (20, 5))
for i in range(2):
    plt.subplot(1, 2, (i + 1))
    plt.title(titles[i])
    plt.plot(history.history[figures[i]])
    plt.plot(history.history['val_{}'.format(figures[i])])
    
plt.show()