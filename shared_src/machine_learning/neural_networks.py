#!/usr/bin/env python3
'''
authors: Chris Kim and Raymond Sutrisno

Model Architecture: 1, Score: 0.828 +/- 0.170
Model Architecture: 2, Score: 0.880 +/- 0.125
Model Architecture: 3, Score: 0.939 +/- 0.100
'''

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
import pandas as pd
import sys
from sklearn.model_selection import StratifiedKFold # class proportions are preserved in each folder
from utils import *

def build_keras_sequential(activation='sigmoid', architecture=(12,5)):
   assert(len(architecture) >= 2)
   model = Sequential()
   input_dim = architecture[0]
   for units in architecture[1:]:
      layer = Dense(units, activation=activation, input_dim=input_dim)
      model.add(layer)
      input_dim = units
      '''
      The optimizer RMSprop divides the gradient by a running average of its recent 
      magnitude. It is usually a good choice for recurrent neural networks. We've seen
      a sharp drop in performance with higher epoch numbers. This problem hasn't been
      observed with the optimizer adam.
      Loss function specific for multi-class classification problems.
      '''
   model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
   return model



def train_keras_sequential(file,config,epoch_num):
   df = pd.read_csv(file)
   data, labels = pre_process_data(file, pickled=False, feature_cols=[], label_col=-1, drop=['file_names'],
                                   one_hot=False, shuffle=True, standard_scale=True)
   print(data)
   print(labels)
   kfold = StratifiedKFold(n_splits=10,random_state=1).split(data,labels)
   scores = []
   c_matrix = np.zeros((2,2))
   for k,(train,test) in enumerate(kfold):
      y = np.eye(len(np.unique(labels)))[labels]
      model = build_keras_sequential(architecture=config)
      model.fit(data[train],y[train],epochs=epoch_num,batch_size=10)
      score = model.evaluate(data[test],y[test],batch_size=10) # score = [test loss, test accuracy]
      scores.append(score[1])
      predictions = model.predict_classes(data[test])
      c_matrix = c_matrix + confusion_matrix(y[test],predictions)
   return np.mean(scores), np.std(scores), c_matrix

def Main(file):
   model_configurations = [
      (12, 5),
      (12, 8, 5),
      (12, 10, 7, 5),
   ]
   epoch = [10,10,10]
   means = []
   stds = []
   cms = []
   for k,config in enumerate(model_configurations):
      mean,std, cm = train_keras_sequential(file, config, epoch[k])
      means.append(mean)
      stds.append(std)
      cms.append(cm)
   for k in range(len(model_configurations)):
      print('Model Architecture:%2d'
            '\tScore: %.3f +/- %.3f\n\t' 
            '\tConfusion Matrix: ' %(k+1, means[k], stds[k]))
      print(cms[k])


if __name__=='__main__' and len(sys.argv) > 1:
   import argparse
   directory = sys.argv[1]
   Main(directory)


## TTD: see whether if model needs to be recompiled with each k iteration so that it starts with random weights. look at fit documentation 
##      part of train_keras_model

## TTD: get confusion matrix, F1 scores as well 