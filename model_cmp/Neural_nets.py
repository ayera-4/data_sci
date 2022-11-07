#load dataset
import pandas as pd
import numpy as np

from sklearn.datasets import load_boston, load_iris
from sklearn.model_selection import train_test_split

X = load_iris().data
y = load_iris().target

#create train, test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#convert labels to one_hot encoding
from sklearn.preprocessing import OneHotEncoder
lb = OneHotEncoder()
lb.fit(y_train)
y_train = lb.transform(y_train)
y_test = lb.transform(y_test)

#Design neural network
from tensorflow import keras
def model_def(input_shape, num_classes):
  #Build the architecture
  model = keras.Sequential(
      [
          keras.layers.Dense(128, activation="relu", input_shape=input_shape),
          keras.layers.Dense(64, activation="relu"),
          keras.layers.Dense(32, activation="relu"),
          keras.layers.Dense(num_classes, activation="softmax")
      ]
  )
  print(model.summary())
  return model

#Train model
num_epochs =50
input_shape = X_train[0].shape
num_classes = y_train.shape[-1]

model = model_def(input_shape, num_classes)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
 
history = model.fit(X_train, y_train, epochs=num_epochs)

score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test Accuracy:", score[1])
