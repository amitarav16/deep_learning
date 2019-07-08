#imports
import pandas as pd 
import numpy as np 
import tensorflow as tf 
from sklearn.model_selection import train_test_split
# from tensorflow import keras 
from keras.models import Sequential
from keras.layers import Dense
#loading data
data = pd.read_csv("data.csv")
data = data.drop(['index'],axis=1)
#input and output data
x = data.loc[:,('area','bathrooms')]
y = data.loc[:,('price')]
#training and testing data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
#creating model
model = Sequential()
n_cols = x_train.shape[1]
#adding layers in models
model.add(Dense(64,activation='relu',input_shape=(n_cols,)))
model.add(Dense(32,activation='relu'))
model.add(Dense(1))
#compilinf model
model.compile(optimizer='adam',loss='mean_squared_error')
#training model
model.fit(x_train,y_train,epochs=100)
#prediction
pred = model.predict(x_test)
print(pred)
print(y_test)