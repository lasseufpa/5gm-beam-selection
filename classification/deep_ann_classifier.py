'''Trains a simple deep NN on ITA paper drop based dataset.

Adapted by AK: Feb 7, 2018 - I took out the graphics. Uses Pedro's datasets with 6 antenna elements per UPA, which has 26 classes.

See for explanation about convnet and filters:
https://datascience.stackexchange.com/questions/16463/what-is-are-the-default-filters-used-by-keras-convolution2d
and
http://cs231n.github.io/convolutional-networks/
'''

from __future__ import print_function

import keras
#from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
#from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adagrad
import numpy as np
from sklearn.preprocessing import minmax_scale
import keras.backend as K
import copy

#one can disable the imports below if not plotting / saving
from keras.utils import plot_model
import matplotlib.pyplot as plt

batch_size = 32
epochs = 50

numUPAAntennaElements=4*4 #4 x 4 UPA
trainFileName = '../datasets/all_train_classification.npz' #(22256, 24, 362)
print("Reading dataset...", trainFileName)
train_cache_file = np.load(trainFileName)

testFileName = '../datasets/all_test_classification.npz' #(22256, 24, 362)
print("Reading dataset...", testFileName)
test_cache_file = np.load(testFileName)

#input features (X_test and X_train) are arrays with matrices. Here we will convert matrices to 1-d array

X_train = train_cache_file['position_matrix_array'] #inputs
y_train = train_cache_file['best_ray_array'] #outputs, one integer for Tx and another for Rx
X_test = test_cache_file['position_matrix_array'] #inputs
y_test = test_cache_file['best_ray_array'] #outputs, one integer for Tx and another for Rx
#best_ray_array could be the array name, if not found, try change it
#y_test = test_cache_file['best_ray_array'] #outputs, 4 angles

#check if data have the correct shape
if len(y_test.shape) == 3:
    y_test_shape = y_test.shape
    X_test_shape = X_test.shape
    X_test =  X_test.reshape((X_test_shape[0]*X_test_shape[1],X_test_shape[2], X_test_shape[3]))
    y_test = y_test.reshape((y_test_shape[0]*y_test_shape[1],2))

if len(y_train.shape) == 3:
    y_train_shape = y_train.shape
    X_train_shape = X_train.shape
    X_train =  X_train.reshape((X_train_shape[0]*X_train_shape[1],X_train_shape[2], X_train_shape[3]))
    y_train = y_train.reshape((y_train_shape[0]*y_train_shape[1],2))

#X_train and X_test have values -4, -3, -1, 0, 2. Simplify it to using only -1 for blockers and 1 for 
X_train[X_train==-4] = -1
X_train[X_train==-3] = -1
X_train[X_train==2] = 1
X_test[X_test==-4] = -1
X_test[X_test==-3] = -1
X_test[X_test==2] = 1

#convert output (i,j) to single number (the class label) and eliminate pairs that do not appear
train_full_y = (y_train[:,0] * numUPAAntennaElements + y_train[:,1]).astype(np.int)
test_full_y = (y_test[:,0] * numUPAAntennaElements + y_test[:,1]).astype(np.int)
train_classes = set(train_full_y) #find unique pairs
test_classes = set(test_full_y) #find unique pairs
classes = train_classes.union(test_classes)

y_train = np.empty(y_train.shape[0])
y_test = np.empty(y_test.shape[0])
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

# convert class vectors to binary class matrices. This is equivalent to using OneHotEncoder
#from sklearn.preprocessing import OneHotEncoder
#encoder = OneHotEncoder()
#y_train = encoder.fit_transform(y_train.reshape(-1, 1))
y_train = keras.utils.to_categorical(y_train, numClasses)
original_y_test = copy.deepcopy(y_test).astype(int)
y_test = keras.utils.to_categorical(y_test, numClasses)

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
#model.add(Dense(numClasses, activation='softmax'))

#model.summary()

#model.add(Dense(1,input_shape=input_shape, activation='relu'))
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
model.add(Dense(numClasses, activation='softmax'))

#model.add(Conv2D(20, kernel_size=(16, 16),
#                 activation='relu',
#				 strides=[1,1],
#				 padding="SAME",
#                 input_shape=input_shape))
#model.add(Conv2D(4, (6, 4), padding="SAME", activation='relu'))
#model.add(Conv2D(16, (10, 2), padding="SAME", activation='relu'))
#model.add(MaxPooling2D(pool_size=(4, 2)))
#model.add(Dropout(0.5))
#model.add(Flatten())
#model.add(Dense(2, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(numClasses, activation='softmax'))

model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
              
# compile model.
#model.compile(loss='mean_squared_error',
#              optimizer=Adagrad(),
#              metrics=['accuracy','mae'])

history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    shuffle=True,
                    validation_split=validationFraction)
                    #validation_data=(X_test, y_test))

# print results
score = model.evaluate(X_test, y_test, verbose=0)
print(model.metrics_names)
#print('Test loss rmse:', np.sqrt(score[0]))
#print('Test accuracy:', score[1])
print(score)

val_acc = history.history['val_acc']
acc = history.history['acc']
f = open('classification_output.txt','w')
f.write('validation_acc\n')
f.write(str(val_acc))
f.write('\ntrain_acc\n')
f.write(str(acc))
f.close()

#enable if want to plot images
if False:
    from keras.utils import plot_model
    import matplotlib.pyplot as plt
    
    #install graphviz: sudo apt-get install graphviz and then pip install related packages
    plot_model(model, to_file='classification_model.png', show_shapes = True)


    pred_test = model.predict(X_test)
    for i in range(len(y_test)):
        if (original_y_test[i] != np.argmax(pred_test[i])):
            myImage = X_test[i].reshape(nrows,ncolumns)
            plt.imshow(myImage)
            plt.show()
            print("Type <ENTER> for next")
            input()
