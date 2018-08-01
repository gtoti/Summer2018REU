#!/usr/bin/env python3
'''
authors: Chris Kim and Raymond Sutrisno

n_splits = 10
epochs = 700 for all architectures

Model Architecture: 1
        Score: 0.744 +/- 0.139
        f1, Class 1: 0.333
        f1, Class 2: 0.000
        f1, Class 3: 0.571
        f1, Class 4: 0.792
        f1, Class 5: 0.824

Model Architecture: 2
        Score: 0.863 +/- 0.134
        f1, Class 1: 1.000
        f1, Class 2: nan
        f1, Class 3: 0.818
        f1, Class 4: 0.846
        f1, Class 5: 0.889

Model Architecture: 3
        Score: 0.845 +/- 0.139
        f1, Class 1: 1.000
        f1, Class 2: nan
        f1, Class 3: 0.762
        f1, Class 4: 0.824
        f1, Class 5: 0.894
'''

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
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
   model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
   return model

def train_keras(data,labels,config,epoch_num):
   n_splits = 10
   con_matrix_labels = sorted(np.unique(labels))
   con_matrix = np.zeros(shape=(len(con_matrix_labels), len(con_matrix_labels)))
   
   kfold = StratifiedKFold(n_splits=n_splits,random_state=1).split(data,labels)
   scores = []

   for k,(train,test) in enumerate(kfold):
      y = np.eye(len(np.unique(labels)))[labels]
      model = build_keras_sequential(architecture=config)
      print('Training Fold',k+1,'\n')
      model.fit(data[train],y[train],epochs=epoch_num,batch_size=10,verbose=1)
      #print('\n\n\n')
      score = model.evaluate(data[test],y[test],batch_size=10,verbose=0) # score = [test loss, test accuracy]
      scores.append(score[1])
      predictions = model.predict_classes(data[test])
      con_matrix = con_matrix + confusion_matrix(labels[test],predictions,labels=con_matrix_labels)
   
   return np.mean(scores), np.std(scores), con_matrix

def Main(file):
   np.seed.random(0)
   
   data, labels = pre_process_data(file, pickled=False, feature_cols=[], label_col=-1, drop=['file_names'],
                                   one_hot=False, shuffle=True, standard_scale=True, index_col=0)
   input_num = data.shape[1]
   output_num = np.unique(labels).shape[0]

   model_configurations = [
      (input_num, output_num),
      (input_num, 8, output_num),
      (input_num, 10, 7, output_num),
   ]
   epoch = [800,800,800]

   means = []
   stds = []
   cms = []
   for k,config in enumerate(model_configurations):
      print('\nModel',k+1,'_____________________________________________\n')
      mean,std, cm = train_keras_sequential(data, labels, config, epoch[k])
      means.append(mean)
      stds.append(std)
      cms.append(cm)
   print('\n\n\n')
   for k in range(len(model_configurations)):
      print('\nModel Architecture:%2d'
            '\n\tScore: %.3f +/- %.3f' %(k+1, means[k], stds[k]))
      print('\tConfusion Matrix:\n', cms[k])
      f1scores = confusion_matrix_f1_scores(cms[k])
      for i,f1 in enumerate(f1scores):
         print('\tf1, Class '+str(i+1)+': %.3f' %(f1))
      f1scores = [x for x in f1scores if str(x) != 'nan']
      print('f1','Mean: %.3f' %np.mean(f1scores))
      df_cm = pd.DataFrame(cms[k], index = [i for i in range(len(cms[k]))],
                   columns = [i for i in range(cms[k].shape[0])])
      plt.figure(figsize = (10,7))
      sn.heatmap(df_cm, annot=True, annot_kws ={"size":20})
      plt.show()


if __name__=='__main__' and len(sys.argv) > 1:
   import argparse
   directory = sys.argv[1]
   Main(directory)

