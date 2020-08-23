# -*- coding: utf-8 -*-
"""
Created on Fri May  8 18:07:11 2020

@author: Shrikant Agrawal
"""

# 1. Data Processing

# Importing the library
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
x=dataset.iloc[:,3:13]
y=dataset.iloc[:,13]

# Create dummy variable
geography= pd.get_dummies(x['Geography'], drop_first=True)
gender = pd.get_dummies(x['Gender'], drop_first=True)

# Concatenate the data frames
x=pd.concat([x,geography, gender], axis=1)

# Drop unnecessary columns
x=x.drop(['Geography', 'Gender'], axis =1)

# Other way to convert categorical to numerical
#x=pd.get_dummies(x, drop_first=True)


# Splitting the dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)

# Feature Scalling
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Now lets make the ANN

# Importing Keras library and packages
import keras
from keras.models import Sequential            # Required for every neural netword
from keras.layers import Dense                 # Required to create hidden layers
from keras.layers import ReLU, LeakyReLU, ELU  # Required for activation function
from keras.layers import Dropout               # Required for regularization parameter when Neural network is very deep

# Initializing the ANN
classifier = Sequential()

# Adding the input layer and first hidden layer
classifier.add(Dense(units = 6, kernel_initializer='he_uniform', activation='relu', input_dim =11 ))

# units is for number of neurons in first hidden layer, kernel_initializer is weight initialization tech
# input_dim is total number of input features

# Adding Second hidden layer
classifier.add(Dense(units=6, kernel_initializer='he_uniform', activation='relu'))

# Adding output layer
classifier.add(Dense(units = 1, kernel_initializer = 'glorot_uniform', activation='sigmoid'))

# To see total number of hidden layers and neuron 
classifier.summary()       # param 72 = 11 features * 6 neurons = 66+6 bias  and same for all

# Comppiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

# Optimizer will reduce the loss value - GD, SGD, Minibatch SGD, adam - most popular these days 
# loss - binary output - binary_crossentropy, multi categorical o/p - category _crossentropy 

# Fitting the ANN to the training set
model_history = classifier.fit(x_train, y_train,validation_split=0.33, batch_size=10,nb_epoch=100)
# validation_split is to test model seperately on test dataset
# batch_size - to reduce the computational power, ram will be pretty much free


# Part 3 - Making the prediction and evaluating the model
y_pred = classifier.predict(x_test)
y_pred = (y_pred>0.5)    # whichever is greater than 0.5 True rest all False

# Confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm= confusion_matrix(y_test, y_pred)
cm

score = accuracy_score(y_test, y_pred)
score

# Summarize history for loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'test'], loc = 'upper left')
plt.show()


