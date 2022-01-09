import tensorflow as tf
import tensorflow_datasets as tfds
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from tensorflow import keras

train, test = tfds.load( 'emnist/balanced', split=['train', 'test'], shuffle_files=True)
df_train = tfds.as_dataframe( train)
df_test = tfds.as_dataframe( test)

def reshape_rot_flip( img):
    img = np.rot90(img, k=-1, axes=(0,1))
    img = np.flip( img, (1))
    return img

x_train = np.stack( [reshape_rot_flip(img) for img in df_train['image']]) / 255
y_train = df_train['label'].to_numpy()
x_test = np.stack( [reshape_rot_flip(img) for img in df_test['image']]) / 255
y_test = df_test['label'].to_numpy()

def plot_curve( epochs, hist, list_of_metrics):
    plt.figure()
    plt.xlabel( 'Epoch')
    plt.ylabel( 'Value')
    
    for m in list_of_metrics:
        x = hist[m]
        plt.plot( epochs[1:], x[1:], label=m)
        
    plt.legend()

from tensorflow import keras
model = keras.models.load_model('C:/Users/lovro/FER/5.semestar/Projekt R/ProjektR_klasifikacija/kod/lovro_modeli/model_8898')
model.evaluate( x=x_test, y=y_test, batch_size=4000)

model.metrics_names
 