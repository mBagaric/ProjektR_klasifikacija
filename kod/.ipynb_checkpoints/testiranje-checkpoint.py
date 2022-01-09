{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "executionInfo": {
     "elapsed": 2496,
     "status": "ok",
     "timestamp": 1635678653365,
     "user": {
      "displayName": "Renato Jurisic",
      "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64",
      "userId": "05301050288504605714"
     },
     "user_tz": -60
    },
    "id": "DVSCJsHxlSxq"
   },
   "outputs": [],
   "source": [
    "# dodajemo sve potrebne importe\n",
    "import tensorflow as tf\n",
    "import tensorflow_datasets as tfds\n",
    "from matplotlib import pyplot as plt\n",
    "import pandas as pd\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "executionInfo": {
     "elapsed": 65034,
     "status": "ok",
     "timestamp": 1635678901592,
     "user": {
      "displayName": "Renato Jurisic",
      "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64",
      "userId": "05301050288504605714"
     },
     "user_tz": -60
    },
    "id": "R43IzhZIlVJP"
   },
   "outputs": [],
   "source": [
    "# ucitavanje podataka\n",
    "train, test = tfds.load( 'emnist/balanced', split=['train', 'test'], shuffle_files=True)\n",
    "df_train = tfds.as_dataframe( train)\n",
    "df_test = tfds.as_dataframe( test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "executionInfo": {
     "elapsed": 58,
     "status": "ok",
     "timestamp": 1635678901595,
     "user": {
      "displayName": "Renato Jurisic",
      "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64",
      "userId": "05301050288504605714"
     },
     "user_tz": -60
    },
    "id": "ObVXoVvQlWic"
   },
   "outputs": [],
   "source": [
    "# inicijalno je dataset rotiran za 90 stupnjeva i zrcaljen, mozemo tranirati i na tome, ali mozemo ih i vratiti u originalan oblik\n",
    "def reshape_rot_flip( img):\n",
    "    img = np.rot90(img, k=-1, axes=(0,1))\n",
    "    img = np.flip( img, (1))\n",
    "    return img"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "executionInfo": {
     "elapsed": 5488,
     "status": "ok",
     "timestamp": 1635678907027,
     "user": {
      "displayName": "Renato Jurisic",
      "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64",
      "userId": "05301050288504605714"
     },
     "user_tz": -60
    },
    "id": "By4HmOJ6lX14"
   },
   "outputs": [],
   "source": [
    "# vracanje podataka u originalan oblik\n",
    "x_train = np.stack( [reshape_rot_flip(img) for img in df_train['image']]) / 255\n",
    "y_train = df_train['label'].to_numpy()\n",
    "x_test = np.stack( [reshape_rot_flip(img) for img in df_test['image']]) / 255\n",
    "y_test = df_test['label'].to_numpy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "executionInfo": {
     "elapsed": 50,
     "status": "ok",
     "timestamp": 1635678907029,
     "user": {
      "displayName": "Renato Jurisic",
      "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64",
      "userId": "05301050288504605714"
     },
     "user_tz": -60
    },
    "id": "m1PUxNpvlZM_"
   },
   "outputs": [],
   "source": [
    "# funkcija za crtanje\n",
    "def plot_curve( epochs, hist, list_of_metrics):\n",
    "    plt.figure()\n",
    "    plt.xlabel( 'Epoch')\n",
    "    plt.ylabel( 'Value')\n",
    "    \n",
    "    for m in list_of_metrics:\n",
    "        x = hist[m]\n",
    "        plt.plot( epochs[1:], x[1:], label=m)\n",
    "        \n",
    "    plt.legend()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "executionInfo": {
     "elapsed": 365,
     "status": "ok",
     "timestamp": 1635678907346,
     "user": {
      "displayName": "Renato Jurisic",
      "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64",
      "userId": "05301050288504605714"
     },
     "user_tz": -60
    },
    "id": "-aDiQj58lbyh"
   },
   "outputs": [],
   "source": [
    "# definiranje arhitekture modela\n",
    "def create_model( learning_rate):\n",
    "    model = tf.keras.models.Sequential()\n",
    "    \n",
    "    model.add( tf.keras.layers.Conv2D( filters=32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))\n",
    "    model.add( tf.keras.layers.MaxPool2D( pool_size=(2,2), strides=2))\n",
    "    model.add( tf.keras.layers.Conv2D( filters=64, kernel_size=(3,3), activation='relu', padding='same'))\n",
    "    model.add( tf.keras.layers.MaxPool2D( pool_size=(2,2), strides=2))\n",
    "    model.add( tf.keras.layers.Conv2D( filters=128, kernel_size=(3,3), activation='relu', padding='valid'))\n",
    "    model.add( tf.keras.layers.MaxPool2D( pool_size=(2,2), strides=2))\n",
    "    model.add( tf.keras.layers.Flatten())\n",
    "    model.add( tf.keras.layers.Dense( 64, activation='relu'))\n",
    "    model.add( tf.keras.layers.Dense( 128, activation='relu'))\n",
    "    model.add( tf.keras.layers.Dropout( rate=0.2))\n",
    "    model.add( tf.keras.layers.Dense( 47, activation='softmax'))\n",
    "    \n",
    "    model.compile( optimizer=tf.keras.optimizers.Adam( learning_rate=learning_rate),\n",
    "                    loss=\"sparse_categorical_crossentropy\",\n",
    "                    metrics=['accuracy'])\n",
    "    \n",
    "    return model\n",
    "\n",
    "# funkcija koja trenira model\n",
    "def train_model( model, train_features, train_label, epochs,\n",
    "                    batch_size=None, validation_split=0.1):\n",
    "\n",
    "    history = model.fit( x=train_features, y=train_label, batch_size=batch_size,\n",
    "                           epochs=epochs, shuffle=True,\n",
    "                           validation_split=validation_split)\n",
    "    \n",
    "    epochs = history.epoch\n",
    "    hist = pd.DataFrame( history.history)\n",
    "    \n",
    "    return epochs, hist"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 1000
    },
    "executionInfo": {
     "elapsed": 265360,
     "status": "ok",
     "timestamp": 1635679172700,
     "user": {
      "displayName": "Renato Jurisic",
      "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64",
      "userId": "05301050288504605714"
     },
     "user_tz": -60
    },
    "id": "4JtMCc-Rlcm5",
    "outputId": "e7dd7d1f-2e1a-4644-b786-cc9988b7e288"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/50\n"
     ]
    }
   ],
   "source": [
    "\n",
    "# inicijalizacija nekih hyperparametara\n",
    "learning_rate = 0.001\n",
    "epochs = 50\n",
    "batch_size = 4000\n",
    "validation_split = 0.2\n",
    "\n",
    "# stvori model\n",
    "my_model = create_model( learning_rate)\n",
    "  \n",
    "# pokreni treniranje\n",
    "epochs, hist = train_model( my_model, x_train, y_train,\n",
    "                                  epochs, batch_size, validation_split)\n",
    "\n",
    "# nacrtaj graf preciznosti po epohama\n",
    "list_of_metrics_to_plot = ['accuracy']\n",
    "plot_curve( epochs, hist, list_of_metrics_to_plot)\n",
    "\n",
    "# evaluiraj model na test set-u\n",
    "print(\"\\n Evaluate the new model against the test set:\")\n",
    "my_model.evaluate( x=x_test, y=y_test, batch_size=batch_size)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 252,
     "status": "ok",
     "timestamp": 1635679212897,
     "user": {
      "displayName": "Renato Jurisic",
      "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64",
      "userId": "05301050288504605714"
     },
     "user_tz": -60
    },
    "id": "gIKmNZqOE98e",
    "outputId": "f106d56e-ef4e-4a13-86f2-b79af6f449bd"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['loss', 'accuracy']"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "my_model.metrics_names"
   ]
  }
 ],
 "metadata": {
  "colab": {
   "authorship_tag": "ABX9TyMdzvavMwtUw4ct/kmyQ4dc",
   "collapsed_sections": [],
   "mount_file_id": "1Yq6M19V6kLsNWTQ7K79PxMhjpwUO0qZC",
   "name": "EMINST_classification.ipynb",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
