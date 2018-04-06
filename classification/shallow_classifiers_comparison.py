#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=====================
Classifier comparison
=====================
From:
http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

Adapted by AK: Feb 7, 2018 - I took out the graphics. Uses Pedro's datasets with 6 antenna elements per UPA.

A comparison of a several classifiers in scikit-learn.
"""
print(__doc__)


# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler and by Aldebaro Klautau
# License: BSD 3 clause

import numpy as np

#enable if want to plot images:
#import matplotlib
#matplotlib.use('WebAgg') 
#matplotlib.use('Qt5Agg') 
#matplotlib.use('agg') 
#matplotlib.inline()
#import matplotlib.pyplot as plt

#from matplotlib.colors import ListedColormap
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

names = [ #"Naive Bayes", 
         "Decision Tree", "Random Forest", 
         "AdaBoost",
         "Linear SVM", "RBF SVM", "Gaussian Process",
         "Neural Net", 
         "QDA", "Nearest Neighbors"]

classifiers = [
    #GaussianNB(),
    DecisionTreeClassifier(max_depth=100),
    RandomForestClassifier(max_depth=100, n_estimators=30, max_features=20),
    AdaBoostClassifier(),
    LinearSVC(C=10, loss="hinge"), #linear SVM (maximum margin perceptron)
    SVC(gamma=1, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    MLPClassifier(alpha=1),
    QuadraticDiscriminantAnalysis(),
    KNeighborsClassifier(3)]

numUPAAntennaElements=4*4 #4 x 4 UPA
#trainFileName = '../datasets/all_train_classification.npz' #(22256, 24, 362)
trainFileName = '../datasets/nlos_train_classification.npz' #(22256, 24, 362)
print("Reading dataset...", trainFileName)
train_cache_file = np.load(trainFileName)

#testFileName = '../datasets/all_test_classification.npz' #(22256, 24, 362)
testFileName = '../datasets/nlos_test_classification.npz' #(22256, 24, 362)
print("Reading dataset...", testFileName)
test_cache_file = np.load(testFileName)

#input features (X_test and X_train) are arrays with matrices. Here we will convert matrices to 1-d array

X_train = train_cache_file['position_matrix_array'] #inputs
train_best_tx_rx_array = train_cache_file['best_tx_rx_array'] #outputs, one integer for Tx and another for Rx
X_test = test_cache_file['position_matrix_array'] #inputs
test_best_tx_rx_array = test_cache_file['best_tx_rx_array'] #outputs, one integer for Tx and another for Rx
#print(position_matrix_array.shape)
#print(best_tx_rx_array.shape)

#X_train and X_test have values -4, -3, -1, 0, 2. Simplify it to using only -1 for blockers and 1 for 
X_train[X_train==-4] = -1
X_train[X_train==-3] = -1
X_train[X_train==2] = 1
X_test[X_test==-4] = -1
X_test[X_test==-3] = -1
X_test[X_test==2] = 1

#convert output (i,j) to single number (the class label) and eliminate pairs that do not appear
train_full_y = (train_best_tx_rx_array[:,0] * numUPAAntennaElements + train_best_tx_rx_array[:,1]).astype(np.int)
test_full_y = (test_best_tx_rx_array[:,0] * numUPAAntennaElements + test_best_tx_rx_array[:,1]).astype(np.int)
train_classes = set(train_full_y) #find unique pairs
test_classes = set(test_full_y) #find unique pairs
classes = train_classes.union(test_classes)

y_train = np.empty(train_best_tx_rx_array.shape[0])
y_test = np.empty(test_best_tx_rx_array.shape[0])
for idx, cl in enumerate(classes): #map in single index, cl is the original class number, idx is its index
    cl_idx = np.nonzero(train_full_y == cl)
    y_train[cl_idx] = idx
    cl_idx = np.nonzero(test_full_y == cl)
    y_test[cl_idx] = idx

#newclasses = set(y)
numClasses = len(classes) #total number of labels

train_nexamples=len(X_train)
test_nexamples=len(X_test)
nrows=len(X_train[0])
ncolumns=len(X_train[0][0])

print('test_nexamples = ', test_nexamples)
print('train_nexamples = ', train_nexamples)
print('input matrices size = ', nrows, ' x ', ncolumns)
print('numClasses = ', numClasses)

#convert matrix into 1-d array
X_train = X_train.reshape(train_nexamples,nrows*ncolumns)
X_test = X_test.reshape(test_nexamples,nrows*ncolumns)

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
print(X_test.shape[0]+X_train.shape[0], 'total samples')
print("Finished reading datasets")

# iterate over classifiers
for name, model in zip(names, classifiers):
    print("#### Training classifier ", name)
    model.fit(X_train, y_train)
    print('\nPrediction accuracy for the test dataset')
    pred_test = model.predict(X_test)
    print('{:.2%}\n'.format(metrics.accuracy_score(y_test, pred_test)))
    #now with the train set
    pred_train = model.predict(X_train)
    print('\nPrediction accuracy for the train dataset')
    print('{:.2%}\n'.format(metrics.accuracy_score(y_train, pred_train)))

#enable if want to plot images
#    for i in range(len(y_test)):
#        if (y_test[i] != pred_test[i]):
#            myImage = X_test[i].reshape(nrows,ncolumns)
#            plt.imshow(myImage)
#            plt.show()
#            print("Type <ENTER> for next")
#            input()
