#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import tree

from utils import pre_process_data
from utils import kFold_Scikit_Model_Trainer
from utils import confusion_matrix_f1_scores

def Main():
   import sys
   file_name = sys.argv[1]

   features, labels = pre_process_data(
                        file_name, pickled=False, label_col=-1,
                        drop=["file_names"],
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

