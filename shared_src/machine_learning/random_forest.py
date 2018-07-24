#!/usr/bin/env python3

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as np

from utils import pre_process_data

def train_random_forest(features, labels, kfold_splits=10, return_scores=False):
   kfold = StratifiedKFold(n_splits=kfold_splits, random_state=1).split(features,labels)
   scores = []
   for k, (train,test) in enumerate(kfold):
      clf = RandomForestClassifier()
      clf.fit(features[train], labels[train])
      score = clf.score(features[test], labels[test])
      scores.append(score)

   kfold_accuracy = np.mean(np.array(scores))

   return (kfold_accuracy, scores) if return_scores else kfold_accuracy


def Main():
   '''
   import argparse
   parser = argparse.ArgumentParser()
   '''

   import sys
   file_name = sys.argv[1]

   features, labels = pre_process_data(file_name, pickled=False, label_col=-1, drop='file_names', shuffle=True)
   kfold_splits = 10
   kfold, scores = train_random_forest(features, labels, kfold_splits=10, return_scores=True)

   print('KFold Cross Validation at k={} :'.format(kfold_splits))

   for k, s in enumerate(scores):
      print('Fold: {}, Validation Accuracy: {}'.format(k, s))

   print('mean: {},  std: {}'.format(kfold, np.std(np.array(scores))))

if __name__=='__main__': Main()

