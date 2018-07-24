#!/usr/bin/env python3

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import sys
from sklearn.model_selection import StratifiedKFold # class proportions are preserved in each fold

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
      magnitude. It is usually a good choice for recurrent neural networks.
      Loss function specific for multi-class classification problems.
      '''
   model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
   return model


def train_keras_sequential(file, model):
   df = pd.read_csv(file)
   labels = df['labels'].values
   # labels = np.eye(df['labels'].unique().shape[0])[labels]
   # labels = np.eye(len(np.unique(labels)))[labels]
   data = df.drop(columns=['Unnamed: 0','labels','file_names']).values
   
   kfold = StratifiedKFold(n_splits=10,random_state=1).split(data,labels)
   scores = []
   for k,(train,test) in enumerate(kfold):
      y = np.eye(len(np.unique(labels)))[labels]
      model.fit(data[train],y[train],epochs=50,batch_size=10)
      score = model.evaluate(data[test],y[test],batch_size=10) # score = [test loss, test accuracy]
      scores.append(score[1])
      print('Fold: %2d, Class dist.: %s, Acc: %.3f' %(k+1, np.bincount(labels[train]), score[1]))
   #score = model.evaluate()
   print('\nCV accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

def Main(file):
   model_configurations = [
      (12, 5),
      (12, 8, 5),
      (12, 10, 7, 5),
   ]
   for config in model_configurations:
      model = train_keras_sequential(file, build_keras_sequential(architecture=config))

if __name__=='__main__' and len(sys.argv) > 1:
   import argparse
   directory = sys.argv[1]
   Main(directory)


   