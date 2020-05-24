import numpy as np
import matplotlib.pyplot as mat
import pandas as pd

dataset = pd.read_csv('Social_Network_Ads.csv')

X = dataset.iloc[:,2:4].values
Y = dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

from sklearn.svm import SVC 
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

# Applying K-Fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = 10)
accuracies.mean()
accuracies.std()

# Applying the Grid Search
from sklearn.model_selection import GridSearchCV
parameters = [{'C':[1, 10 ,100, 1000], 'kernel':['linear']},
              {'C': [1, 10, 100, 1000], 'kernel':['rbf'], 'gamma':[0.5, 0.1,0.2, 0.3, 0.6, 0.7, 0.8, 0.9]}]
gridSearchCV = GridSearchCV(estimator = classifier, param_grid = parameters, 
                            scoring = 'accuracy', cv  = 10, n_jobs = -1)
gridSearchCV = gridSearchCV.fit(X_train, Y_train)
best_accuracy = gridSearchCV.best_score_
best_parameters = gridSearchCV.best_params_


from matplotlib.colors import ListedColormap
X_set, Y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min()-1, stop = X_set[:,0].max()+1, step = 0.01), 
                     np.arange(start = X_set[:,1].min()-1, stop = X_set[:,1].max()+1, step = 0.01))
mat.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
mat.xlim(X1.min(), X2.max())
mat.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(Y_set)):
    mat.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1], 
                c = ListedColormap(('red','green'))(i), label = j)
mat.title('Kernel SVM (Training set)')
mat.xlabel('age')
mat.ylabel('Estimated salary')
mat.show()

from matplotlib.colors import ListedColormap
X_set, Y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min()-1, stop = X_set[:,0].max()+1, step = 0.01), 
                     np.arange(start = X_set[:,1].min()-1, stop = X_set[:,1].max()+1, step = 0.01))
mat.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
mat.xlim(X1.min(), X2.max())
mat.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(Y_set)):
    mat.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1], 
                c = ListedColormap(('red','green'))(i), label = j)
mat.title('Kernel SVM (Test set)')
mat.xlabel('age')
mat.ylabel('Estimated salary')
mat.show()