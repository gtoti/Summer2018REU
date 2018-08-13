
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold,GridSearchCV,learning_curve
from sklearn.metrics import confusion_matrix
import sys
from utils import *
from matplotlib import pyplot as plt
import numpy as np

def Main(file_nms, n_trials, k_folds, print_best,
            rand_forest_parameters, print_results, seed):

   param_keys = list(rand_forest_parameters.keys())
   for file_nm in file_nms:
      np.random.seed(seed)

      features, labels = pre_process_data(
            file_nm, pickled=False, feature_cols=[], label_col=-1,
            drop=['file_names'], one_hot=False, shuffle=True,
            standard_scale=False, index_col=0)

      i = len(labels) * 8 // 10
      features_test  = features[i:]
      labels_test    = labels[i:]
      features_train = features[:i]
      labels_train   = labels[:i]

      if 0 in labels:
         if all(labels_test != 0): # ensure at least one class zero
            loc = np.where(labels_train == 0)[0][0]
            labels_train[loc], labels_test[0] = labels_test[0], labels_train[loc]
            features_train[loc], features_test[0] = features_test[0], features_train[loc]
         elif dict(zip(*np.unique(labels_test, return_counts=True)))[0] > 1:
            i = np.where(labels_test == 0)[0][0]
            j = np.where(labels_train != 0)[0][0]
            labels_train[j], labels_test[i] = labels_test[i], labels_train[j]

      np.random.seed() # reseed randomly

      clfs = [GridSearchCV(RandomForestClassifier(), rand_forest_parameters,
              cv=k_folds,return_train_score=True, n_jobs=4)\
              .fit(features_train,labels_train) for i in range(n_trials)]

      results = [(c.best_params_,c.best_score_,\
                  c.score(features_test, labels_test)) for c in clfs]

      results = sorted(results, key=lambda c:c[1], reverse=True)

      if print_results:
         CV_stats = np.array(list(c[1] for c in results))
         CV_stats = np.mean(CV_stats), np.std(CV_stats)
         print("\n\n\n\nResults for :", file_nm)
         print('GridSearch CV_stats -> mean:{:.4f}, std:{:.4f}'.format(*CV_stats))

         if print_best:
            print('{}      CV_score={:.4f}      Test_score={:.4f}'.
                                                format(*results[0]))
         else:
            print('\n'.join(
                'Best Params={}      CV_score={:.4f}      Test_score={:.4f}'\
                .format(*c) for c in results))


      # make learning curves
      best_model_param = results[0][0]
      generate_learning_curve(
         file_nm, best_model_param, features_train, labels_train,
         features_test, labels_test, seed=seed)


      # evaluate final accuracy
      kfoldCV, test_score, con_matrix, test_con_matrix =\
                     evaluate_final_accuracy(
                     file_nm, best_model_param, features_train, labels_train,
                     features_test, labels_test, kfolds=k_folds, seed=seed)


      print('LC -> CV(mean:{:.4f},std:{:.4f}), test_score(mean:{:.4f},std:{:.4f})'\
            .format(*kfoldCV, *test_score))

      print('LC -> confusion_matrix:\n', con_matrix)
      print('LC -> confusion_matrix testing:\n', test_con_matrix)

      F1_scores = confusion_matrix_f1_scores(con_matrix)
      print('LC -> F1:\n', F1_scores)

# end of def Main


def evaluate_final_accuracy(file_nm, best_params, features_train, labels_train,
                            features_test, labels_test, kfolds=10, seed=None):

   np.random.seed(seed)

   test_score = []
   con_matrix_labels = sorted(np.unique(np.append(labels_train, labels_test)))
   con_matrix = np.zeros(shape=(len(con_matrix_labels), len(con_matrix_labels)))

   test_con_matrix = np.zeros(shape=(len(con_matrix_labels), len(con_matrix_labels)))

   def model_callback(model, train, test):
      nonlocal test_score
      nonlocal con_matrix
      test_score.append(model.score(features_test, labels_test))

      y_ = labels_train[test]
      y = model.predict(features_train[test])
      con_matrix += confusion_matrix(y_, y, labels=con_matrix_labels)

   kfoldCV= kFold_Scikit_Model_Trainer(
                        features_train, labels_train,
                        lambda:RandomForestClassifier(**best_params, n_jobs=4),
                        return_scores=True, kfold_splits=kfolds,
                        model_callback=model_callback)

   test_score = np.mean(test_score), np.std(test_score)
   kfoldCV = kfoldCV[0], np.std(kfoldCV[1])

   full_model = RandomForestClassifier(**best_params, n_jobs=4).\
                     fit(features_train, labels_train)
   y_ = labels_test
   y = full_model.predict(features_test)
   test_con_matrix = confusion_matrix(y_, y, labels=con_matrix_labels)

   return kfoldCV, test_score, con_matrix, test_con_matrix


def generate_learning_curve(file_nm, best_params, features_train, labels_train,
                            features_test, labels_test, seed=None):

   model = RandomForestClassifier(**best_params, n_jobs=4)

   np.random.seed(seed)
   train_sizes, train_scores, cv_scores = learning_curve(
                     estimator=model,
                     X=features_train, y=labels_train,
                     train_sizes=np.linspace(0.1, 1, 10),
                     cv=10, n_jobs=4)

   train_mean = np.mean(train_scores, axis=1)
   train_std  = np.std(train_scores, axis=1)
   cv_mean    = np.mean(cv_scores, axis=1)
   cv_std     = np.std(cv_scores, axis=1)

   plt.plot(train_sizes, train_mean, color='blue', marker='o',
                      markersize=5, label='training accuracy')

   plt.fill_between(train_sizes, train_mean+train_std, train_mean-train_std,
                    alpha=0.15, color='blue')

   plt.plot(train_sizes, cv_mean, color='green', marker='s', linestyle='--',
                      markersize=5, label='validation accuracy')

   plt.fill_between(train_sizes, cv_mean+cv_std, cv_mean-cv_std,
                    alpha=0.15, color='green')

   plt.grid()
   plt.xlabel('Number of Training Examples')
   plt.ylabel('Accuracy')
   plt.legend(loc='lower right')
   plt.ylim([np.min(cv_mean-cv_std), 1.0])
   plt.title('LC: {},\nseed={}'.format(str(best_params), seed))
   output_name = input()
   plt.savefig(output_name, dpi=300)

# end of def generate_learning_curve


if __name__=='__main__':
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

   parser.add_argument('-n','--n_trials', type=int, action='store', default=10,\
   help=
   """ 
   Number of trials to perform gridsearch on RandomForest parameters.
   Default N=10.
   """, dest="TRIALS")

   parser.add_argument('-k','--kfoldnum', type=int, action='store', default=10,\
   help=
   """
   K number to perform K fold cross validation. Default k=10
   """, dest="KFOLDNUM")

   parser.add_argument('-p','--print_params', action='store_true',default=False,
   help=\
   """
   This flag prints the parameters being tested in GridSearch for
   random forest. Exits once printed.
   """, dest="PRINT_TOGGLE")

   parser.add_argument('-P','--print', action='store_true',default=False,
   help=\
   """
   If this flag is on, print parameter results from trials.
   """, dest="PRINT")

   parser.add_argument('-B','--best', action='store_true',default=False,
   help=\
   """
   If this flag is on, only the best parameter is printed out of all trials.
   """, dest="BEST")

   parser.add_argument('--seed', action='store',type=int,default=None,
   help=\
   """
   Seed for random number generator. Seed is set once before reading the file.
   """, dest="SEED")

   parser.add_argument('-D', '--depth', action='store',type=int,default=None,
   help=\
   """
   Prune Depth
   """, dest="DEPTH")


   arguments = parser.parse_args()

   file_nms = arguments.data_set
   print_params = arguments.PRINT_TOGGLE
   n_trials = arguments.TRIALS
   k_folds = arguments.KFOLDNUM
   print_best = arguments.BEST
   print_results = arguments.PRINT
   seed = arguments.SEED
   prune_lvl = arguments.DEPTH

   rand_forest_parameters = {
      'n_estimators':(10,15,20,25,50,100),
      'max_features':(None, 'auto'),
      #'criterion':('gini','entropy')
      #'max_depth':(2,3,4,5,6,7,8,9,10)
      'max_depth':(prune_lvl,)
   }

   if print_params:
      print(rand_forest_parameters)
      quit()

   if len(file_nms) == 0:
      print("Nothing to process!")
      quit()

   # -------------+ VVV +---+ DO STUFF +---+ VVV +------------- #

   assert(n_trials >= 1)
   assert(k_folds >= 2)
   Main(file_nms, n_trials, k_folds, print_best,
         rand_forest_parameters, print_results, seed)


