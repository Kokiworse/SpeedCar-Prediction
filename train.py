import numpy as np
import cv2
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Flatten, Dense, Dropout, TimeDistributed
from decimal import *
import matplotlib.pyplot as plt
from keras.models import load_model

TXT = "C:\\Users\\koki\\Desktop\\python\\comma\\speedchallenge\\data\\train.txt"
img_rows = 150
img_cols = 150
timesteps = 16

def model():
    model = Sequential()
    input_shape=(timesteps, img_rows, img_cols, 1)

    # 1st layer group
    model.add(TimeDistributed(Conv2D(32, (3, 3),  activation='relu', padding='same', name='conv1', input_shape=input_shape, kernel_initializer='glorot_uniform')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding='valid', name='pool1')))

    # 2nd layer group
    model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='pool2')))

    # 3rd layer group
    model.add(TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3a')))
    model.add(TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3b')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='pool3')))

    # 4th layer group
    model.add(TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4a')))
    model.add(TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4b')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='pool4')))

    # 5th layer group
    model.add(TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same', name='conv5a')))
    model.add(TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same', name='conv5b')))
    model.add(TimeDistributed(ZeroPadding2D(padding=((1, 1)), name='zeropad5')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='pool5')))
    model.add(TimeDistributed(Flatten()))

    model.add(Dense(512, activation='relu', name='fc6'))
    model.add(Dropout(.4))
    model.add(Dense(512, activation='relu', name='fc7'))
    model.add(Dropout(.4))
    model.add(Dense(1, activation='linear', name='fc8'))

    return model

def readFromTxt():
    getcontext().prec = 10
    with open(TXT) as f:
        content = f.readlines()
    content = [Decimal(x.strip()) for x in content]
    return np.array(content)

def normalize(X):
    X = np.array(X, dtype=np.float64)
    X -= np.mean(X)
    X /= np.std(X, axis = 0)
    return X

if __name__ == "__main__":
    X = np.load('dataClean100gray.npz')
    X = X['arr_0']
    #normalize
    X = normalize(X)
    print((X).shape)
    model = model()
    y = readFromTxt()[ : len(X)]
    print(len(X), len(y))
    batch_size = X.shape[0] // timesteps
    X = X.reshape(batch_size, timesteps, img_rows, img_cols, 1)
    y = y.reshape(batch_size, timesteps, 1)
    print(y.shape)
    print(X.shape)
    model.compile(optimizer='adam', loss='mse')
    history = model.fit(X, y, validation_split=0.1, batch_size=8, shuffle=True, epochs=50, verbose=1, initial_epoch=0, steps_per_epoch=None, validation_steps=None)
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    model.save('CNN-50epochs.h5')
    print("Model saved")
