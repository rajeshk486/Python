#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 02:36:46 2017

@author: hadoop
"""

import pandas as pd
import numpy as np

import matplotlib.mlab as mlab 
import matplotlib.pyplot as plt 

from sklearn import tree 
from sklearn.tree import DecisionTreeClassifier 
from sklearn import metrics

#reading dataset
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:,:-1].values
y=dataset.iloc[:,4].values


#missing data:- no missing data

#categorical dataset
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:,3] = labelencoder_x.fit_transform(x[:,3])

onehotencoder = OneHotEncoder(categorical_features = [3])
x = onehotencoder.fit_transform(x).toarray()
