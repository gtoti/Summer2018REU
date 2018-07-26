from sklearn import svm
import sys
from utils import *
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

def build_and_train_SVM(file, model):
	features, labels = pre_process_data(file, pickled=False, feature_cols=[], label_col=-1, drop=['file_names'], one_hot=False, 
										shuffle=True, standard_scale=True)
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
                    model_constructor=lambda: model,
                    kfold_splits=kfold_splits,
                    return_scores=True,
                    model_callback=model_callback)
	print('Score: %.3f +/- %.3f' %(kfold,np.std(scores)))
	print('Confusion Matrix:\n', con_matrix)
	f1scores = confusion_matrix_f1_scores(con_matrix)
	for k,f1 in enumerate(f1scores):
		print('f1, Class '+str(k+1)+': %.3f' %(f1))
	f1scores = [x for x in f1scores if str(x) != 'nan']
	print('f1','AVG: %.3f' %np.mean(f1scores))
	df_cm = pd.DataFrame(con_matrix, index = [i for i in range(len(con_matrix_labels))],
						 columns = [i for i in range(len(con_matrix_labels))])
	plt.figure(figsize = (10,7))
	sn.heatmap(df_cm, annot=True, annot_kws ={"size":20})
	#plt.show()

def Main(file):
	print('\n\tSVC with Linear Kernel:')
	build_and_train_SVM(file, svm.SVC(kernel='linear', C=1, gamma=1))
	print('\n\tSVC with RBF Kernel: ')
	build_and_train_SVM(file, svm.SVC(kernel='rbf', C=1, gamma=1))
	print('\n\tLinearSVC: ')
	build_and_train_SVM(file, svm.LinearSVC(C=1))

if __name__== '__main__' and len(sys.argv) > 1:
	import argparse
	directory = sys.argv[1]
	Main(directory)
	


