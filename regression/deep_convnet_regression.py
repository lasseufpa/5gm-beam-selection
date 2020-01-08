'''Trains a deep NN on ITA paper drop-based regression dataset.

Adapted by AK: Feb 7, 2018 - I took out the graphics. Uses Pedro's datasets with 
regression problem version 6. Problem (regression) is: find the 4 angles of the strong ray

See for explanation about convnet and filters:
https://datascience.stackexchange.com/questions/16463/what-is-are-the-default-filters-used-by-keras-convolution2d
and
http://cs231n.github.io/convolutional-networks/
'''

from __future__ import print_function

import numpy as np
import copy

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
#from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Conv2D, MaxPooling2D #because input is a matrix, I will use 2D
from keras.optimizers import Adagrad
from sklearn.preprocessing import minmax_scale
#import keras.backend as K
from sklearn.preprocessing import MinMaxScaler

#one can disable the imports below if not plotting / saving
from keras.utils import plot_model
import matplotlib.pyplot as plt
#import sys
#np.set_printoptions(threshold=sys.maxsize)
#for neural net training
batch_size = 32
epochs = 500

trainFileName = '../datasets/all_train_regression.npz' #(22256, 24, 362)
print("Reading dataset...", trainFileName)
train_cache_file = np.load(trainFileName)

testFileName = '../datasets/all_test_regression.npz' #(22256, 24, 362)
print("Reading dataset...", testFileName)
test_cache_file = np.load(testFileName)

#input features (X_test and X_train) are arrays with matrices. Here we will convert matrices to 1-d array

X_train = train_cache_file['position_matrix_array'] #inputs
y_train = train_cache_file['best_ray_array'] #outputs, 4 angles
X_test = test_cache_file['position_matrix_array'] #inputs
y_test = test_cache_file['best_ray_array'] #outputs, 4 angles

if len(y_test.shape) == 3:
    y_test_shape = y_test.shape
    X_test_shape = X_test.shape
    X_test =  X_test.reshape((X_test_shape[0]*X_test_shape[1],X_test_shape[2], X_test_shape[3]))
    y_test = y_test.reshape((y_test_shape[0]*y_test_shape[1],4))

if len(y_train.shape) == 3:
    y_train_shape = y_train.shape
    X_train_shape = X_train.shape
    X_train =  X_train.reshape((X_train_shape[0]*X_train_shape[1],X_train_shape[2], X_train_shape[3]))
    y_train = y_train.reshape((y_train_shape[0]*y_train_shape[1],4))

#X_train and X_test have values -4, -3, -1, 0, 2. Simplify it to using only -1 for blockers and 1 for 
X_train[X_train==-4] = -1
X_train[X_train==-3] = -1
X_train[X_train==2] = 1
X_test[X_test==-4] = -1
X_test[X_test==-3] = -1
X_test[X_test==2] = 1

#Regression may work better with scaled data
#Scale data to range -1,1 for each feature
scaler=MinMaxScaler(copy=True, feature_range=(-1, 1)) #get one scaling for each of the 4 angles
#scaler=MinMaxScaler(copy=True, feature_range=(0, 1)) #get one scaling for each of the 4 angles
scaler.fit(y_train)
print(scaler)
y_train=scaler.transform(y_train)
ynonscaled_test=copy.deepcopy(y_test) #keep original data
y_test=scaler.transform(y_test)

train_nexamples=len(X_train)
test_nexamples=len(X_test)
nrows=len(X_train[0])
ncolumns=len(X_train[0][0])
numOutputs=len(y_test[0])

print('test_nexamples = ', test_nexamples)
print('train_nexamples = ', train_nexamples)
print('input matrices size = ', nrows, ' x ', ncolumns)
print('num. outputs = ', numOutputs)

#here, do not convert matrix into 1-d array
#X_train = X_train.reshape(train_nexamples,nrows*ncolumns)
#X_test = X_test.reshape(test_nexamples,nrows*ncolumns)

#fraction to be used for training set
validationFraction = 0.2 #from 0 to 1

#Keras is requiring an extra dimension: I will add it with reshape
X_train = X_train.reshape(X_train.shape[0], nrows, ncolumns, 1)
X_test = X_test.reshape(X_test.shape[0], nrows, ncolumns, 1)
input_shape = (nrows, ncolumns, 1) #the input matrix with the extra dimension requested by Keras

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
print(X_test.shape[0]+X_train.shape[0], 'total samples')
print("Finished reading datasets")

#do not convert class vectors to binary class matrices (used for classification with NN)
#y_train = keras.utils.to_categorical(y_train, numOutputs)
#y_test = keras.utils.to_categorical(y_test, numOutputs)

# declare model Convnet with two conv1D layers following by MaxPooling layer, and two dense layers
# Dropout layer consists in randomly setting a fraction rate of input units to 0 at each update during training time, which helps prevent overfitting.

model = Sequential()

#model.add(Conv2D(24, kernel_size=(2, 2), activation='relu', padding='same', input_shape=(135,2)))
#model.add(MaxPooling1D(3))
#model.add(Conv2D(16, kernel_size=(2, 2), activation='relu', padding='same'))
#model.add(MaxPooling1D(3))
#model.add(Dropout(0.5))
#model.add(Dense(8, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Flatten())
#model.add(Dense(numOutputs, activation='softmax'))

model.add(Conv2D(100, kernel_size=(10,10),
            activation='relu',
#				 strides=[1,1],
#				 padding="SAME",
             input_shape=input_shape))
model.add(Conv2D(50, (12, 12), padding="SAME", activation='relu'))
model.add(MaxPooling2D(pool_size=(6, 6)))
model.add(Conv2D(20, (10, 10), padding="SAME", activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(4, activation='relu'))
#model.add(Dropout(0.3))
model.add(Flatten())
#model.add(Activation('tanh'))
#model.add(Activation('softmax')) #softmax for probability
#model.add(Dense(numClasses, activation='softmax'))


#model.add(Dense(4,input_shape=input_shape, activation='relu'))
#model.add(Conv2D(10, kernel_size=(2,2),
#                 activation='relu',
#				 strides=[1,1],
#				 padding="SAME",
#                 input_shape=input_shape))
#model.add(Conv2D(10, (6, 6), padding="SAME", activation='relu'))
#model.add(Conv2D(10, (10, 10), padding="SAME", activation='relu'))
#model.add(MaxPooling2D(pool_size=(4, 2)))
#model.add(Dropout(0.3))
#model.add(Flatten())
#model.add(Dense(10, activation='relu'))
#model.add(Dropout(0.3))
#model.add(Activation('tanh'))
#model.add(Activation('softmax')) #softmax for probability
model.add(Dense(numOutputs, activation='linear'))
#model.add(Dense(numOutputs, activation='relu'))

model.summary()

#look for optimizers at
#https://keras.io/optimizers/

#look at metrics at
#https://machinelearningmastery.com/custom-metrics-deep-learning-keras-python/
model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adadelta(),
              #optimizer=keras.optimizers.Adam(),
              metrics=['mape'])
              #metrics=['mse', 'mae', 'mape', 'cosine'])

if False:
    #install graphviz: sudo apt-get install graphviz and then pip install related packages
    plot_model(model, to_file='model.png', show_shapes = True)

#model.compile(loss=keras.losses.categorical_crossentropy,
#              optimizer=keras.optimizers.Adadelta(),
#              metrics=['accuracy'])

# compile model.=
#model.compile(loss='mean_squared_error',
#              optimizer=Adagrad(),
#              metrics=['accuracy','mae'])

history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    shuffle=True,
                    validation_split=0.1)
                    #validation_data=(X_test, y_test))

yscaled_test_predicted = model.predict(X_test)
ynonscaled_test_predicted = scaler.inverse_transform(yscaled_test_predicted)
errorHere = ynonscaled_test - ynonscaled_test_predicted
mses = np.mean( errorHere * errorHere, 0 )
print('Total root MSE for all angles = ', np.sqrt(np.mean(mses)))
print('RMSE for each angle (De,Da,Ae,Aa) = ', np.sqrt(mses))

# print results
score = model.evaluate(X_test, y_test, verbose=1)
print(model.metrics_names)
#print('Test loss rmse:', np.sqrt(score[0]))
#print('Test accuracy:', score[1])
print(score)
print(history.history.keys())

mapes = history.history['mape']
f = open('regression_output.txt','w')
f.write(str(mapes))
f.close()
    
#plt.plot(history.history['mean_squared_error'])
#plt.plot(history.history['mean_absolute_error'])
#plt.plot(history.history['mean_absolute_percentage_error'])
#plt.plot(history.history['cosine_proximity'])
#plt.show()

exit()
if False: #enable if want to plot angles
        print(departure_angle[best_p_idx])
        print(arrival_angle[best_p_idx])
        ylim = np.min((departure_ele, arrival_ele)) - 2, \
            np.max((departure_ele, arrival_ele)) + 2
        ax = plt.subplot(121, polar=True)
        ax.plot([0, np.deg2rad(departure_azi)], [ylim[0], departure_ele], 'b',
             label='departure')
    #    plt.hold(True)
        ax.plot([0, np.deg2rad(arrival_azi)], [ylim[0], arrival_ele], 'r',
             label='arrival')
        ax.set_ylim(*ylim)
        ax.legend()
        ax = plt.subplot(122)
        ax.imshow(position_matrix.T, origin='lower')
        plt.show()
