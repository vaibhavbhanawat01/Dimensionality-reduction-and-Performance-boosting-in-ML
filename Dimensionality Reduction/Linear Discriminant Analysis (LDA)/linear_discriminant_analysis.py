# -*- coding: utf-8 -*-
"""
Created on Sun May 17 18:26:26 2020

@author: vaibhav_bhanawat
"""

import pandas as pd
import matplotlib.pyplot as mat
import numpy as np

dataset = pd.read_csv('Wine.csv')

X = dataset.iloc[:, 0:13].values
Y = dataset.iloc[:, 13].values

# Split of Data into training and testing 
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 0)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test) 

# Dimensionality reduction
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components = 2)
X_train = lda.fit_transform(X_train, Y_train)
X_test = lda.transform(X_test)

# Logistic regression
from sklearn.linear_model import LogisticRegression
regression = LogisticRegression(random_state = 0)
regression.fit(X_train, Y_train)

Y_pred = regression.predict(X_test)

# Confusion metrics
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
