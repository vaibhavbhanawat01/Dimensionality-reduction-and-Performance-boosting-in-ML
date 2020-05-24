# -*- coding: utf-8 -*-
"""
Created on Sat May 16 18:16:16 2020

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
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
variance = pca.explained_variance_ratio_

# Logistic regression
from sklearn.linear_model import LogisticRegression
regression = LogisticRegression(random_state = 0)
regression.fit(X_train, Y_train)

Y_pred = regression.predict(X_test)

# Confusion metrics
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

# from matplotlib.colors import ListedColormap
# X_set, Y_set = X_train, Y_train
# X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min()-1, stop = X_set[:,0].max()+1, step = 0.01), 
#                      np.arange(start = X_set[:,1].min()-1, stop = X_set[:,1].max()+1, step = 0.01))
# mat.contourf(X1, X2, regression.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#              alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
# mat.xlim(X1.min(), X2.max())
# mat.ylim(X2.min(), X2.max())

# for i, j in enumerate(np.unique(Y_set)):
#     mat.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1], 
#                 c = np.array(['red','green', 'blue']).reshape(-1, 1)(i), label = j)
# mat.title('Logistic regression (Training set)')
# mat.xlabel('PC1')
# mat.ylabel('PC2')
# mat.show()

# from matplotlib.colors import ListedColormap
# X_set, Y_set = X_test, Y_test
# X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min()-1, stop = X_set[:,0].max()+1, step = 0.01), 
#                      np.arange(start = X_set[:,1].min()-1, stop = X_set[:,1].max()+1, step = 0.01))
# mat.contourf(X1, X2, regression.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#              alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
# mat.xlim(X1.min(), X2.max())
# mat.ylim(X2.min(), X2.max())

# for i, j in enumerate(np.unique(Y_set)):
#     mat.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1], 
#                 c = ListedColormap(np.array(['green','red', 'blue']))(i), label = j)
# mat.title('Logistic regression (Test set)')
# mat.xlabel('PC1')
# mat.ylabel('PC2')
# mat.show()








