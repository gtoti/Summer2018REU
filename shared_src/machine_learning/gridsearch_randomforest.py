from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import sys
from utils import *

def Main(file):
	parameters = {'n_estimators':(5,10,15,16,17,18,19,20,21,22,23,24,25,30,35,40,45,50), 'max_features':(None, 'auto'), 'criterion':('gini','entropy')}
	model = RandomForestClassifier()
	clf = GridSearchCV(model,parameters,cv=10,return_train_score=True)
	data, labels = pre_process_data(file, pickled=False, feature_cols=[], label_col=-1, drop=['file_names'],
                                   one_hot=False, shuffle=True, standard_scale=True, index_col=0)
	i = len(labels) * 9//10
	data_train = data[:i]
	data_test = data[i:]
	labels_train = labels[:i]
	labels_test = labels[i:]
	clf.fit(data_train,labels_train)
	print("Best Parameter:",clf.best_params_)


if __name__=='__main__' and len(sys.argv) > 1:
   import argparse
   directory = sys.argv[1]
   Main(directory)
