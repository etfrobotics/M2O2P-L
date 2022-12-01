import sklearn.preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

DATA_ADDRESS = r'D:\PyCharm Projekti\Leap Motion projekat\Leap Motion python 3.8'
MODEL_ADDRESS = r'D:\PyCharm Projekti\Leap Motion projekat\Leap Motion python 3.8'

class Data:
    def __init__(self):
        pass
    def create_data(self, tr_ts):
        """
        This function assumes that there is a train and test folder in MODEL_ADDRESS folder:
            MODEL_ADDRESS
                Train
                Test
        And those folders contain folders with images that belong to different classes.
        Finally, it creates numpy arrays that will be used for training and testing the model later.

        :param tr_ts: if we want to create train set -  'train', if we want to create test set - 'test'
        """
        x_data = []
        y_data = []
        for u, person in enumerate(os.listdir(DATA_ADDRESS+'/' + tr_ts + '/')): # go through persons
            for klasa, j in enumerate(os.listdir(DATA_ADDRESS+'/' + tr_ts + '/' + person + '/')):  # go through classes
                for k in os.listdir(DATA_ADDRESS+'/' + tr_ts +'/' + person + '/' + j + '/'):  # go through images
                    img = Image.open(DATA_ADDRESS+'/' + tr_ts + '/' + person +'/' + j + '/' + k).convert('L')  # load and convert to black and white
                    img = img.resize((320, 120))  # rescale to one size
                    arr = np.array(img)  # convert to numpy array
                    mask3d = np.tile(arr[:, :, None], [1, 1, 3])
                    x_data.append(mask3d)
                    y_data.append(klasa)
            c = list(zip(x_data, y_data))
            #random.shuffle(c)

        x_data, y_data = zip(*c)
        x_data = np.array(x_data, dtype='float32')
        x_data = x_data / 255
        y_data = np.array(y_data)
        y_data = y_data.reshape(y_data.shape[0], 1)

        enc = sklearn.preprocessing.OneHotEncoder().fit(y_data) # one hot encoding of output array
        y_data = enc.transform(y_data).toarray()

        np.save(DATA_ADDRESS + '/' + tr_ts + 'x.npy', x_data)
        np.save(DATA_ADDRESS + '/' + tr_ts + 'y.npy', y_data)


class Model:

    def __init__(self):
        """
        We assume that there are allready created numpy arrays with data called trainx.npy, trainy.npy, testx.npy and
        testy.npy in DATA_ADDRESS folder. It created the vgg16 model and loads training and testing data.
        """
        self.model = self.create_vgg16()
        self.model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        #x_data = np.load('trainx.npy')
        #y_data = np.load('trainy.npy')
        self.x_test = np.load('testx.npy')
        self.y_test = np.load('testy.npy')
        #self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=42, shuffle=True)

    def create_vgg16(self):
        """
        This function creates vgg16 model with a final layer for classification of 4 hand gestures.
        """
        base_model = tf.keras.applications.vgg16.VGG16(input_shape=(120,320,3),  # Shape of our images
                       include_top=False,  # Leave out the last fully connected layer
                       weights='imagenet')
        for layer in base_model.layers:
            layer.trainable = False
        x = tf.keras.layers.Flatten()(base_model.output)

        # Add a fully connected layer with 512 hidden units and ReLU activation
        x = tf.keras.layers.Dense(512, activation='relu',kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4))(x)

        # Add a dropout rate of 0.5
        x = tf.keras.layers.Dropout(0.2)(x)

        # Add a final sigmoid layer with 1 node for classification output
        x = tf.keras.layers.Dense(4, activation='sigmoid',kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4))(x)

        model = tf.keras.models.Model(base_model.input, x)
        model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
                      # Loss function to minimize
                      loss='binary_crossentropy', # tf.keras.losses.SparseCategoricalCrossentropy(),
                      # List of metrics to monitor
                      metrics=['acc', tf.keras.metrics.SparseCategoricalAccuracy()],
                      )

        return model

    def train_model(self):
        """
        This function trains a vgg16 model on training data, with regularization and validation.
        """
        history = self.model.fit(self.x_train, self.y_train, epochs=5, batch_size=64, verbose=1, validation_data=(self.x_val, self.y_val))
        self.model.save(MODEL_ADDRESS+'\model_vgg_new.h5')
        print(history)

    def load_model(self, model_name):
        self.model.load_weights(MODEL_ADDRESS+r'\\'+model_name )

    def test_model(self, model_name):
        """
        This function validates the performances of the model that has allready been trained and creates a confusion
        matrix.
        """
        self.load_model(model_name)
        b = self.model.predict(self.x_test)
        c = np.zeros(b.shape)
        for i in range(len(b)):
            k = np.argmax(b[i, :])
            c[i, k] = 1
        mat = sklearn.metrics.confusion_matrix(np.argmax(self.y_test, axis=1), np.argmax(c, axis=1))

        df_cm = pd.DataFrame(mat, ['fist', 'hand', 'L', 'vertical'], ['fist', 'hand', 'L', 'vertical'])
        sn.set(font_scale=1.4)  # for label size
        sn.heatmap(df_cm, annot=True)  # font size

        plt.xlabel('Predicted class')
        plt.ylabel('Real class')
        plt.title('Hand gesture classification with VGG16')
        plt.show()

        y_real = np.argmax(self.y_test, axis=1)
        y_predicted = np.argmax(c, axis=1)
        print('Accuracy is: '+str((y_real==y_predicted).sum()/y_real.shape[0]))

from PIL import Image
if __name__=='__main__':
    #data = Data()
    #data.create_data('Test')

    #Mreza = Model()
    #Mreza.train_model()
    #Mreza.test_model('model_vgg4.h5')
    # Acc 98.25%

    #im1 = Image.open(r'C:\Users\Jelena\Desktop\slika337.jpg')
    #im2 = Image.open(r'C:\Users\Jelena\Desktop\slika400.jpg')
    #im3 = Image.open(r'C:\Users\Jelena\Desktop\slike\slika89.jpg')
    #im4 = Image.open(r'C:\Users\Jelena\Desktop\slike\slika221.jpg')
    """
    figure, ax = plt.subplots(nrows=2, ncols=1, figsize=(70, 50), dpi=80)
    ax = ax.ravel()
    im0 = ax[0].imshow(im1, cmap='gray')
    ax[0].set_title('Fist mistaken for L symbol', fontsize=20), ax[0].axis('off')
    im1 = ax[1].imshow(im2, cmap='gray')
    ax[1].set_title('L symbol mistaken for first', fontsize=20), ax[1].axis('off')

    plt.show()
    """
    mat = np.array([[427,17,44,12],[0,496,4,0],[6,19,474,1],[32,0,40,428]])

    df_cm = pd.DataFrame(mat, ['fist', 'hand', 'L', 'vertical'], ['fist', 'hand', 'L', 'vertical'])
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True, cmap = "Blues", fmt='g')  # font size

    plt.xlabel('Predicted class')
    plt.ylabel('Real class')
    plt.title('Hand gesture classification with VGG16')
    plt.show()
