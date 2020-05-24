# -*- coding: utf-8 -*-
"""
Created on Mon May 18 21:57:51 2020

@author: vaibhav_bhanawat
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as mat

dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_country = LabelEncoder()
label_gender = LabelEncoder()
X[:, 1] = label_country.fit_transform(X[:, 1])
X[:, 2] = label_gender.fit_transform(X[:, 2])

from sklearn.compose import ColumnTransformer
columnTran = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [1])], 
                               remainder = 'passthrough')
X = np.array(columnTran.fit_transform(X))
X = X[:, 1:]

# Train and test data split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, 
                                                    random_state = 0)

from xgboost import XGBClassifier