# pip install fastai==0.7.0 --no-deps
# pip install torch==0.4.1 torchvision==0.2.1

# from fastai.conv_learner import *
# from fastai.dataset import *

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import scipy.optimize as opt
import matplotlib.pyplot as plt

DATA_PATH = '/Users/tawate/Documents/H2O_Analytics/data/Kaggle/Human Atlas Protein Location/'
TRAIN = DATA_PATH + 'train/'
TEST = DATA_PATH + 'test/'
LABELS = DATA_PATH + 'train.csv'
SPLIT = DATA_PATH + 'protein-trainval-split/'
nw = 2 #number of workers for dataloader

# label dictionary
name_label_dict = {
0:  'Nucleoplasm',
1:  'Nuclear membrane',
2:  'Nucleoli',   
3:  'Nucleoli fibrillar center',
4:  'Nuclear speckles',
5:  'Nuclear bodies',
6:  'Endoplasmic reticulum',   
7:  'Golgi apparatus',
8:  'Peroxisomes',
9:  'Endosomes',
10:  'Lysosomes',
11:  'Intermediate filaments',
12:  'Actin filaments',
13:  'Focal adhesion sites',   
14:  'Microtubules',
15:  'Microtubule ends',  
16:  'Cytokinetic bridge',   
17:  'Mitotic spindle',
18:  'Microtubule organizing center',  
19:  'Centrosome',
20:  'Lipid droplets',
21:  'Plasma membrane',   
22:  'Cell junctions', 
23:  'Mitochondria',
24:  'Aggresome',
25:  'Cytosol',
26:  'Cytoplasmic bodies',   
27:  'Rods & rings' }
# write the below tables to a comma separated text file
test_names = sorted({f[:36] for f in os.listdir(TEST)})
train_names = sorted({f[:36] for f in os.listdir(TRAIN)})
train_names = train_names[1:]

# Load Data and Read into Train and Validation Text files
with open(SPLIT + 'tr_names.txt' , 'r') as text_file:
    tr_n = text_file.read().split(',')

with open(SPLIT + 'val_names.txt', 'r') as text_file:
    val_n = text_file.read().split(',')

val_n = sorted({f[:36] for f in os.listdir(TEST)})
tr_n = sorted({f[:36] for f in os.listdir(TRAIN)})
tr_n = train_names[1:]
print(len(tr_n), len(val_n))

# Create duplicates for rare classes in train set (Class)
class Oversampling:
    def __init__(self,path):
        self.train_labels = pd.read_csv(path).set_index('Id')
        self.train_labels['Target'] = [[int(i) for i in s.split()]
                                       for s in self.train_labels['Target']]
        
        # set the min number of duplicates for each class
        self.multi = [1,1,1,1,1,1,1,1
                      ,4,4,4,1,1,1,1,4
                      ,1,1,1,1,2,1,1,1
                      ,1,1,1,4]
        
        
    
    def get(self, image_id):
        labels = self.train_labels.loc(image_id, 'Target') if image_id in self.train_labels.index else []
        m = 1
        for l in labels:
            if m < self.multi[l]: 
                m = self.multi[l]
        return m

s = Oversampling(LABELS)
tr_n = [idx for idx in tr_n for _ in range(s.get(idx))]
print(len(tr_n),flush=True)