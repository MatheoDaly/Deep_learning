#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 09:00:12 2018

@author: matheo
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt









######################################### PREPARATION OF DATAS ###############################
datas = pd.read_csv("/home/matheo/Code/Jupyter Notebook/Kaggle - titanic/train.csv")
X = datas.iloc[:,[2,4,5,9]].values
y = datas.iloc[:, 1].values



###Take care of missing values for the variable Age 
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 2:3])
X[:, 2:3] = imputer.transform(X[:, 2:3])




from sklearn.preprocessing import LabelEncoder
###Let's encode all of the X_train variables in numbers 
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#♣Changement d'échelle 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)







########################################## TRAIN THE NEURAL NETWORK ################################

####Import Keras
import keras 
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import Dropout

#Initialisation 
classifier = Sequential()


#Add an entry and a fold 
classifier.add(Dense(units = 3, activation = "relu", 
                     kernel_initializer = "uniform", input_dim = 4))
classifier.add(Dropout(rate = 0.1))

#Add a second fold
classifier.add(Dense(units = 3, activation = "relu", 
                     kernel_initializer = "uniform"))
classifier.add(Dropout(rate = 0.1))

#Add the exit 
classifier.add(Dense(units = 1, activation = "sigmoid", 
                     kernel_initializer = "uniform"))

#compiler
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", 
                   metrics = ["accuracy"])

#Training
classifier.fit(X_train, y_train, batch_size =20, epochs = 500)



y_pred = classifier.predict(X_test)
y_pred = (y_pred> 0.5)

#Matrice de confusion
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)






################################ CROSS VALIDATION ######################################
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier(): 
    classifier = Sequential()
    ##Add an entry and a fold 
    classifier.add(Dense(units = 3, activation = "relu", 
                     kernel_initializer = "uniform", input_dim = 4))
    classifier.add(Dropout(rate = 0.1))
    #Add a second fold
    classifier.add(Dense(units = 3, activation = "relu", 
                     kernel_initializer = "uniform"))
    classifier.add(Dropout(rate = 0.1))
    #Add the exit 
    classifier.add(Dense(units = 1, activation = "sigmoid", 
                     kernel_initializer = "uniform"))
    #compiler
    classifier.compile(optimizer = "adam", loss = "binary_crossentropy", 
                   metrics = ["accuracy"])
    return classifier 

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
precision = cross_val_score (estimator = classifier, X=X_train, y = y_train, cv = 10)


###Moyenne de la validation croisée
mean = precision.mean()

standard_deviation = precision.std()