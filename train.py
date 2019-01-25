'''
TODOs:
- Implement drop of random pixels for image augmenting
- Normalization because dataset is too large and it breaks
'''


import numpy as np
import cv2
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Flatten, Dense, Dropout, TimeDistributed
from decimal import *
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.optimizers import SGD
from sklearn.preprocessing import StandardScaler

TXT = "C:\\Users\\koki\\Desktop\\python\\comma\\speedchallenge\\data\\train.txt"
img_rows = 150
img_cols = 150
timesteps = 16
number_pixels = 150 * 150
percent = round(number_pixels * 0.1)

def model():
    model = Sequential()
    input_shape=(timesteps, img_rows, img_cols, 3)

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

    # FC
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
    # for i in range(len(X)):
    #     X = X / 255
    return X

def augment(image):
    rdm = np.random.randint(101)
    #random flip
    if rdm > 80:
        image = cv2.flip(image, 1)
    #brightness change
    if rdm < 20:
        bright = 0.5 + np.random.random(1)[0] * 0.5
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv[:,:,2] = hsv[:,:,2] * bright
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    #drop random pixels
    if rdm > 40 and rdm < 60:
        # for i in range(img_rows):
        #     for j in range(img_cols):
        pass
    return image

if __name__ == "__main__":
    X = np.load('trainRGB.npz')
    X = X['arr_0']
    print(X.shape)
    c = 0
    print("Normalized")
    for i in range(len(X)):
        rdm = np.random.randint(101)
        if rdm < 25:
            X[i] = augment(X[i])
            c += 1
    print("%d images changed" %c)
    model = model()
    y = readFromTxt()[ : len(X)]
    print(len(X), len(y))
    batch_size = X.shape[0] // timesteps
    X = X.reshape(batch_size, timesteps, img_rows, img_cols, 3)
    y = y.reshape(batch_size, timesteps, 1)
    # sgd = SGD(lr=1e-5, decay=0.0005, momentum=0.9)
    model.compile(optimizer='adam', loss='mse')
    history = model.fit(X, y, validation_split=0.1, batch_size=8, shuffle=True, epochs=20, verbose=1, initial_epoch=0, steps_per_epoch=None, validation_steps=None)
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    model.save('CNN-20epochs.h5')
    print("Model saved")
