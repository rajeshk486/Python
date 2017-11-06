# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
dataset= pd.read_csv('Data.csv')
X= dataset.iloc[:,:-1].values
Y =dataset.iloc[:,3].values

#handling missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values ='NaN',strategy = 'mean',axis =0) # you can use median and most_frequent
imputer = imputer.fit(X[:,1:3])
X[:,1:3]= imputer.transform(X[:,1:3])

#encoding the categorical data
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X= onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y= labelencoder_Y.fit_transform(Y)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
