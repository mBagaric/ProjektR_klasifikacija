# dodajemo sve potrebne importe
import tensorflow as tf
import tensorflow_datasets as tfds
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

import os

# ucitavanje podataka
train, test = tfds.load( 'emnist/balanced', split=['train', 'test'], shuffle_files=True)
df_train = tfds.as_dataframe( train)
df_test = tfds.as_dataframe( test)

# inicijalno je dataset rotiran za 90 stupnjeva i zrcaljen, mozemo tranirati i na tome, ali mozemo ih i vratiti u originalan oblik
def reshape_rot_flip( img):
    img = np.rot90(img, k=-1, axes=(0,1))
    img = np.flip( img, (1))
    return img

# vracanje podataka u originalan oblik
x_train = np.stack( [reshape_rot_flip(img) for img in df_train['image']]) / 255
y_train = df_train['label'].to_numpy()
x_test = np.stack( [reshape_rot_flip(img) for img in df_test['image']]) / 255
y_test = df_test['label'].to_numpy()

# funkcija za crtanje
def plot_curve( epochs, hist, list_of_metrics):
    plt.figure()
    plt.xlabel( 'Epoch')
    plt.ylabel( 'Value')
    
    for m in list_of_metrics:
        x = hist[m]
        plt.plot( epochs[1:], x[1:], label=m)
        
    plt.legend()
    #plt.show()
    
# definiranje arhitekture modela
def create_model( learning_rate):
    model = tf.keras.models.Sequential()
    
    model.add( tf.keras.layers.Conv2D( filters=32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))
    model.add( tf.keras.layers.MaxPool2D( pool_size=(2,2), strides=2))
    model.add( tf.keras.layers.Conv2D( filters=64, kernel_size=(3,3), activation='relu', padding='same'))
    model.add( tf.keras.layers.MaxPool2D( pool_size=(2,2), strides=2))
    model.add( tf.keras.layers.Conv2D( filters=128, kernel_size=(3,3), activation='relu', padding='valid'))
    model.add( tf.keras.layers.MaxPool2D( pool_size=(2,2), strides=2))
    model.add( tf.keras.layers.Flatten())
    model.add( tf.keras.layers.Dropout( rate=0.2))
    #model.add( tf.keras.layers.Dense( 784, activation='relu'))
    #model.add( tf.keras.layers.Dense( 512, activation='relu'))
    model.add( tf.keras.layers.Dense( 256, activation='relu'))
    model.add( tf.keras.layers.Dropout( rate=0.1))
    model.add( tf.keras.layers.Dense( 256, activation='relu'))
    model.add( tf.keras.layers.Dense( 256, activation='relu'))
    model.add( tf.keras.layers.Dropout( rate=0.1))
    model.add( tf.keras.layers.Dense( 47, activation='softmax'))
    
    model.compile( optimizer=tf.keras.optimizers.Adam( learning_rate=learning_rate),
                    loss="sparse_categorical_crossentropy",
                    metrics=['accuracy'])
    
    return model

# funkcija koja trenira model
def train_model( model, train_features, train_label, epochs,
                    batch_size=None, validation_split=0.1):

    history = model.fit( x=train_features, y=train_label, batch_size=batch_size,
                           epochs=epochs, shuffle=True,
                           validation_split=validation_split,)

    model.save("lovro_modeli/model_1")
    epochs = history.epoch
    hist = pd.DataFrame( history.history)
    
    return epochs, hist


# inicijalizacija nekih hyperparametara
learning_rate = 0.001
epochs = 100
batch_size = 5000
validation_split = 0.2

# stvori model
my_model = create_model( learning_rate)

# pokreni treniranje
epochs, hist = train_model( my_model, x_train, y_train, epochs, batch_size, validation_split)

# nacrtaj graf preciznosti po epohama
list_of_metrics_to_plot = ['accuracy']
plot_curve( epochs, hist, list_of_metrics_to_plot)

# evaluiraj model na test set-u
print("\n Evaluate the new model against the test set:")
my_model.evaluate( x=x_test, y=y_test, batch_size=batch_size)

my_model.metrics_names
