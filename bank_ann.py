# -*- coding: utf-8 -*-
"""
Created on Fri May 15 18:30:00 2020

@author: 18175
"""


# DATA PREPROCESSING

# importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# import dataset

dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values


# encoding categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x1 = LabelEncoder()
x[:, 1] = labelencoder_x1.fit_transform(x[:, 1])
labelencoder_x2 = LabelEncoder()
x[:, 2] = labelencoder_x2.fit_transform(x[:, 2])
# create dummy variables for Geography 
onehotencoder = OneHotEncoder(categorical_features = [1])
x = onehotencoder.fit_transform(x).toarray()
x = x[:, 1:]


# splitting dataset into training and test sets

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# feature scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)



# MAKING THE ANN

# import keras libraries and packages

import keras
from keras.models import Sequential
from keras.layers import Dense


# initializing the ann

classifier = Sequential()


# adding the input layer and the first hidden layer

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
# update to: classifier.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform"))


# adding the second hidden layer

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))


# adding the output layer

classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
# update to: classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))


# compiling the ann

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# fitting the ann to the training set

classifier.fit(x_train, y_train, batch_size = 10, epochs = 100)



# PREDICTIONS AND EVALUATING THE MODEL

# predicting the test set results

y_pred = classifier.predict(x_test)
# converting churn probability to T/F using a threshold of 0.5
y_pred = (y_pred > 0.5)


# making the confusion matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# compute accuracy
(1505 + 213)/2000
# the accuracy is 0.859


