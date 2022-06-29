import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras import optimizers
import random
import os
#print(os.listdir("../drive"))
print(os.listdir("AutismDataset"))


filenames = os.listdir("AutismDataset/train")
categories = []
for filename in filenames:
    #print("opudj")
    category = filename.split('.')[0]
    if category == 'Autistic':
        categories.append(str(1))
    else:
        categories.append(str(0))

train_df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})
print(train_df)

