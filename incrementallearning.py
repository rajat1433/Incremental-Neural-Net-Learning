# -*- coding: utf-8 -*-
"""
@author: Rajat
"""

import glob
import os
import gzip
import sys
import pickle
from tqdm import trange
from functools import reduce
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import keras
from keras.datasets import mnist
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

#==============================================================================
#                             Data Pre Processing
#==============================================================================


Thedataset = pd.read_csv('Preprocessed2.csv')

X = Thedataset.iloc[:, 2:12].values
y = Thedataset.iloc[:, 12].values

X1=X[:10000,:]
X2=X[10000:20000,:]
X3=X[20000:30000,:]
X4=X[30000:,:]

y1=y[:10000]
y2=y[10000:20000]
y3=y[20000:30000]
y4=y[30000:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train) 
y_test = sc_y.transform(y_test)


model = Sequential()
model.add(Dense(20, input_dim=10, kernel_initializer='normal', activation='relu'))
model.add(Dense(18, kernel_initializer='normal', activation='relu'))
model.add(Dense(18, kernel_initializer='normal', activation='relu'))
model.add(Dense(18, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')

# evaluate model with standardized dataset
model.fit(X_train,y_train,nb_epoch=100, batch_size=5,verbose=1)

# Save a model you have trained
json_string = model.to_json() 
open('model_incremental_architecture.json', 'w').write(json_string) 
model.save_weights('model_incremental_weights.h5') 


#---------------------------------------------------------------------------------

#Now we got more training data

# Load the model
model = model_from_json(open('model_incremental_architecture.json').read()) 
model.load_weights('model_incremental_weights.h5')
model.compile(loss='mean_squared_error', optimizer='adam')

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train) 
y_test = sc_y.transform(y_test)


model.fit(X_train,y_train,nb_epoch=100, batch_size=5,verbose=1)


json_string = model.to_json() 
open('model2_incremental_architecture.json', 'w').write(json_string) 
model.save_weights('model2_incremental_weights.h5') 



#------------------------------------------------------------------------------

#Now we got more training data

model = model_from_json(open('model2_incremental_architecture.json').read()) 
model.load_weights('model2_incremental_weights.h5')
model.compile(loss='mean_squared_error', optimizer='adam')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X3, y3, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train) 
y_test = sc_y.transform(y_test)


model.fit(X_train,y_train,nb_epoch=100, batch_size=5,verbose=1)

json_string = model.to_json() 
open('model3_incremental_architecture.json', 'w').write(json_string) 
model.save_weights('model3_incremental_weights.h5') 




#------------------------------------------------------------------------------

#Now we got more training data

model = model_from_json(open('model3_incremental_architecture.json').read()) 
model.load_weights('model3_incremental_weights.h5')
model.compile(loss='mean_squared_error', optimizer='adam')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X4, y4, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train) 
y_test = sc_y.transform(y_test)


model.fit(X_train,y_train,nb_epoch=100, batch_size=5,verbose=1)

json_string = model.to_json() 
open('model4_incremental_architecture.json', 'w').write(json_string) 
model.save_weights('model4_incremental_weights.h5') 


#---------------------------------------------------------------------------------