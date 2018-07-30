
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import sys
from utils import *
import numpy as np

if __name__=='__main__':
   rand_forest_parameters = {
      'n_estimators':(10,15,20,25,50,100),
      'max_features':(None, 'auto'),
      'criterion':('gini','entropy')
   }

   import argparse
   parser = argparse.ArgumentParser(
      description=
      """This python script performs GridSearch on various parameters
      for Random forest on data sets."""
   )

   parser.add_argument('data_set', type=str, nargs='*',default=[],\
   help=\
   """
   The data set in csv form. This script assumes the first column is the
   index column and that there is a column called file_names which it will drop
   before reading.""")

   parser.add_argument('-n','--n_trials', type=int, nargs=1, default=10,\
   help=
   """ 
   Number of trials to perform gridsearch on RandomForest parameters.
   Default N=10.
   """, dest="TRIALS")

   parser.add_argument('-k','--kfoldnum', type=int, nargs=1, default=10,\
   help=
   """
   K number to perform K fold cross validation. Default k=10
   """, dest="KFOLDNUM")


   parser.add_argument('-p','--print_params', action='store_true',default=False,
   help=\
   """
   This flag prints the parameters being tested in GridSearch for
   random forest.""", dest="PRINT_TOGGLE")


   arguments = parser.parse_args()

   file_nms = arguments.data_set
   print_params = arguments.PRINT_TOGGLE
   n_trials = arguments.TRIALS
   k_folds = arguments.KFOLDNUM

   if print_params:
      print(rand_forest_parameters)
      quit()

   if len(file_nms) == 0:
      print("Nothing to processed!")
      quit()


   print("# Running with args:", arguments)

   for file_nm in file_nms:
      features, labels = pre_process_data(
            file_nm, pickled=False, feature_cols=[], label_col=-1,
            drop=['file_names'], one_hot=False, shuffle=True,
            standard_scale=False, index_col=0)

      i = len(labels) * 9//10
      features_train = features[:i]
      features_test  = features[i:]
      labels_train   = labels[:i]
      labels_test    = labels[i:]

      model = RandomForestClassifier()
      clfs=[GridSearchCV(model,rand_forest_parameters, cv=k_folds,\
            return_train_score=True).fit(features_train, labels_train)\
            for i in range(n_trials)]

      results = [(str(c.best_params_),c.best_score_,\
                  c.score(features_test, labels_test)) for c in clfs]

      print("#\n#Results for :", file_nm)
      print('\n'.join('Best Params={}\tCV_score={:.4f}\tTest_score={:.4f}'\
                .format(*c) for c in sorted(results, key=lambda c:c[1])))

      CV_stats = np.array(list(c[1] for c in results))
      CV_stats = np.mean(CV_stats), np.std(CV_stats)
      print('#GridSearch CV_stats -> mean:{:.4f}, std:{:.4f}'.format(*CV_stats))

