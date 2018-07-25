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
import pandas as pd
import sys
from sklearn.model_selection import StratifiedKFold # class proportions are preserved in each folder
# from utils import *

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



def train_keras_sequential(file, model,epoch_num):
   df = pd.read_csv(file)
   labels = df['labels'].values
   data = df.drop(columns=['Unnamed: 0','labels','file_names']).values 
   kfold = StratifiedKFold(n_splits=10,random_state=1).split(data,labels)
   scores = []
   for k,(train,test) in enumerate(kfold):
      y = np.eye(len(np.unique(labels)))[labels]
      model.fit(data[train],y[train],epochs=epoch_num,batch_size=10)
      score = model.evaluate(data[test],y[test],batch_size=10) # score = [test loss, test accuracy]
      scores.append(score[1])
      #print('Fold: %2d, Class dist.: %s, Acc: %.3f' %(k+1, np.bincount(labels[train]), score[1]))
   #print('\nCV accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))
   return np.mean(scores), np.std(scores)

def Main(file):
   model_configurations = [
      (12, 5),
      (12, 8, 5),
      (12, 10, 7, 5),
   ]
   epoch = [1000,1000,1000]
   means = []
   stds = []
   for k,config in enumerate(model_configurations):
      mean,std = train_keras_sequential(file, build_keras_sequential(architecture=config), epoch[k])
      print(mean)
      means.append(mean)
      stds.append(std)
   for k in range(len(model_configurations)):
      print('Model Architecture:%2d, Score: %.3f +/- %.3f' %(k+1, means[k], stds[k]))

if __name__=='__main__' and len(sys.argv) > 1:
   import argparse
   directory = sys.argv[1]
   Main(directory)


## TTD: see whether if model needs to be recompiled with each k iteration so that it starts with random weights. look at fit documentation 
##      part of train_keras_model

## TTD: get confusion matrix, F1 scores as well 