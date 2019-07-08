#import
import pandas as pd 
import numpy as np
import tensorflow as tf  
import matplotlib.pyplot as plt 
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dense
#loading data
fashion_mnist = keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()
train_images = train_images/255.0
test_images = test_images/255.0
#model
model = Sequential()
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dense(10,activation='softmax'))
#compiling model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#training model
model.fit(train_images,train_labels,epochs=10)
#evalutating
model.evaluate(test_images,test_labels)