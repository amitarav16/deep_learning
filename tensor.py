import pandas as pd 
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
#loading data
data = pd.read_csv("data.csv")
#remove the unnecessary data
data = data.drop(['index','price','sq_price'],axis=1)
data = data[:10]
#adding labels in our data
data.loc[:,('y1')]= [1,1,1,0,1,0,1,1,0,0]
data.loc[:,('y2')] = data['y1'] == 0
#true false to 1 & 0
data.loc[:,('y2')] = data.loc[:,('y2')].astype(int)

#convertnig data to tensor
input_x = data.loc[:,('area','bathrooms')].as_matrix()
input_y = data.loc[:,('y1','y2')].as_matrix()
#hyperparameters
learning_rate = 0.000001
training_epochs = 2000
display_steps = 50
n_samples = input_y.size
#placeholder to fed input data
x = tf.placeholder(tf.float32,[None,2])
#weights
w = tf.Variable(tf.zeros([2,2]))
#bias
b = tf.Variable(tf.zeros([2]))
#y = mx+b
y_values = tf.add(tf.matmul(x,w),b)
#applying softmax
y_ = tf.nn.softmax(y_values)
#feed in a matrix of lables
y = tf.placeholder(tf.float32,[None,2])
#training
cost = tf.reduce_sum(tf.pow(y_ - y,2))/(2*n_samples)
#gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)