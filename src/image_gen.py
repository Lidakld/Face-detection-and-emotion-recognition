#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 15:39:45 2018

@author: lidakuang
"""

import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import logging
#%%
path = '/Users/matthewxfz/workspace/kuangkuang/cs512-f18-lida-kuang/Project/data/fer2013.csv'
data = pd.read_csv(path)[['emotion','pixels']]
#%%
emotion = data['emotion'].values
pixels =  data['pixels']
#%%
images = list(pixels.apply(lambda x: x.split(' ')))
images = np.asfarray(images)
emotions = {0: 'angry',
            1: 'disgust',
            2: 'fear',
            3: 'happy',
            4: 'sad',
            5: 'surprise',
            6: 'neutral'}

#%%
outpath = '/Users/matthewxfz/workspace/kuangkuang/cs512-f18-lida-kuang/Project/data/processed_images/'
for i in range(images.shape[1]):
    img = images[i].reshape(48, 48)
    im = Image.fromarray(img).convert('RGB')
    im_name = '{:04d}'.format(i) + '.jpg'
    if i <= images.shape[1] * 0.8:
        usage = 'train'
    else:
        usage = 'validation'
    im_path = Path(outpath, usage, emotions[emotion[i]])
    im_path.mkdir(parents = True, exist_ok = True)
    im.save(Path(im_path, im_name))
logger = logging.getLogger(__name__)
logger.info('making images set from FER2013 CSV file')  
#data_train = data[data['Usage']=='Training']
#data_test  = data[data['Usage']!='Training']
#X_train = data_train['pixels']
#y_train = data_train['emotion']
#X_test = data_test['pixels']
#y_test = data_test['emotion']
#%%