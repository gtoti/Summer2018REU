#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn import tree

from utils import pre_process_data
from utils import kFold_Scikit_Model_Trainer
from utils import confusion_matrix_f1_scores

def PerformTraining(features, labels, model_constructor):
   kfold_splits=10
   con_matrix_labels = sorted(np.unique(labels))
   con_matrix = np.zeros(shape=(len(con_matrix_labels), len(con_matrix_labels)))

   def model_callback(model, train, test):
      y_ = labels[test]
      y = model.predict(features[test])
      nonlocal con_matrix
      con_matrix += confusion_matrix(y_, y, labels=con_matrix_labels)

   kfold, scores = kFold_Scikit_Model_Trainer(
                     features, labels,
                     model_constructor=model_constructor,
                     kfold_splits=kfold_splits,
                     return_scores=True,
                     model_callback=model_callback)

   print('KFold Cross Validation at k={} :'.format(kfold_splits))
   for k, s in enumerate(scores):
      print('Fold: {}, CV Accuracy: {:.2f}'.format(k+1, s))
   print('mean: {:.2f},  std: {:.2f}'.format(kfold, np.std(np.array(scores))))

   df = pd.DataFrame(columns=con_matrix_labels, index=con_matrix_labels)
   for i, column in enumerate(df.columns): df[column] = con_matrix[:, i]
   print('Confusion Matrix:\n', df)

   f1_scores = confusion_matrix_f1_scores(con_matrix)
   print('f1 scores:')
   print('\n'.join('class: {} -> f1_score: {:.2f}'.format(label, score) for label, score in zip(df.columns, scores)))

def Main():
   import sys
   file_name = sys.argv[1]

   features, labels = pre_process_data(
                        file_name, pickled=False, label_col=-1,
                        drop=["file_names"],
                        #drop=["file_names", "grayscale_ent_devN2", "grayscale_std_devN2","grayscale_skew_devN2","saturation_ent_devN2","saturation_std_devN2","saturation_skew_devN2"],
                        shuffle=True, standard_scale=True, index_col=0)

   print('\nRandom Forest(n_estimators=50):')
   PerformTraining(
      features, labels,
      model_constructor=lambda: RandomForestClassifier(n_estimators=50))
   print('\nScikit DecisionTreeClassifier:')
   PerformTraining(
      features, labels,
      model_constructor=tree.DecisionTreeClassifier)

if __name__=='__main__':
   Main()

