#!/usr/bin/env python3

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from utils import *

def grid_search_rand_forest(file_nm):
   features, labels = pre_process_data(
         file_nm, pickled=False, feature_cols=[], label_col=-1,
         drop=['file_names'], one_hot=False, shuffle=True,
         standard_scale=False, index_col=0)

   i = len(labels) * 9//10
   features_train = features[:i]
   features_test = features[i:]
   labels_train = labels[:i]
   labels_test = labels[i:]

   parameters = {
      'n_estimators':(5,10,15,20,25,50,80,100),
      'max_features':(None, 'auto'),
      'criterion':('gini','entropy')
      }

   model = RandomForestClassifier()
   clf = GridSearchCV(model,parameters,cv=10,return_train_score=True)
   clf.fit(features_train, labels_train)

   #print("Best Parameter:", clf.best_params_)
   return clf

if __name__=='__main__':
   import sys
   clfs = [[grid_search_rand_forest(file_nm) for file_nm in sys.argv[1:]] for i in range(10)]
   for i, data_set in enumerate(sys.argv[1:]):
      print('\nResults for ', data_set)
      print('\n'.join('best_params={} , score={:.4f}'.format(str(clf[i].best_params_), clf[i].best_score_) for clf in clfs))

