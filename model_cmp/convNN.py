import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Flatten, 
                                     Dense, Dropout, BatchNormalization, Activation)

# Download the dataset
from keras.datasets import mnist, fashion_mnist
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
# print(X_train.shape)
# print(X_test.shape)

# View example digit
index = np.random.randint(0,60000)
# plt.imshow(X_train[index])
# print(y_train[index])

# Put it into suitable shape
X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)

# Use one hot encoding to covert the labels
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
lb.fit(y_train)
y_train = lb.transform(y_train)
y_test = lb.transform(y_test)
# print(y_train[0])

def define_cnn(input_shape, num_classes):
    # Build the architecture
    model = keras.Sequential()

    # first CONV => RELU => CONV => RELU => POOL layer set
    model.add(Conv2D(64, (2, 2), padding="same",
        input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (2, 2), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(Dropout(0.25))
    # Second CONV => RELU => CONV => RELU => POOL layer set
    model.add(Conv2D(128, (1, 1), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (1, 1), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(Dropout(0.25))
    # first (and only) set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # softmax classifier
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))

    model.summary()
    return model
  
num_epochs = 10
batch_size = 32
input_shape = X_train.shape[1:]
num_classes = y_train.shape[-1]

model = define_cnn(input_shape, num_classes)
model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
# Train and test
H = model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_data=(X_test, y_test))

score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
