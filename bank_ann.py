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

dataset = pd.read_csv("C:/Users/18175/Documents/udemy/deeplearning/Volume 1 - Supervised Deep Learning/Part 1 - Artificial Neural Networks (ANN)/Section 4 - Building an ANN/Churn_Modelling.csv")
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
from keras.layers import Dropout


# initializing the ann

classifier = Sequential()


# adding the input layer and the first hidden layer with dropout

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
# update to: classifier.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform"))
# add dropout to reduce chances of overfitting 
classifier.add(Dropout(rate = 0.1))

# adding the second hidden layer

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
# add dropout to reduce chances of overfitting 
classifier.add(Dropout(rate = 0.1))

# adding the output layer

classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
# update to: classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))


# compiling the ann

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# fitting the ann to the training set

classifier.fit(x_train, y_train, batch_size = 10, epochs = 100)



# PREDICTING & EVALUATING THE MODEL

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


# predicting a single observation with the following characteristics:
# Geography: France
# Credit Score: 600
# Gender: Male
# Age: 40 years old
# Tenure: 3 years
# Balance: $60000
# Number of Products: 2
# Does this customer have a credit card ? Yes
# Is this customer an Active Member: Yes
# Estimated Salary: $50000

new_prediction = classifier.predict(sc.transform(np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)
# the prediction is False, so this customer does not leave the bank



# EVALUATING, IMPROVING, & TUNING THE ANN

# evaluating the ann

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
# function that builds the architecture of the ann
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10, n_jobs = 1)
# compute the mean and variance of the accuracies
mean = accuracies.mean()
variance = accuracies.std()


# improving the ann

# dropout regularization to reduce overfitting if needed
# added dropout code to input layer and hidden layer in lines 73 & 79 above


# tuning the ann

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
# function that builds the architecture of the ann
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
# create a dictionary of hyperparameters that we want to optimize
parameters = {'batch_size': [25, 32],
           'epochs': [100, 500],
           'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(x_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
















