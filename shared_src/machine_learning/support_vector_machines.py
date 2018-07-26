from sklearn import svm
import sys
from utils import *
from sklearn.metrics import confusion_matrix


def build_and_train_SVM(file):
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
                    model_constructor=lambda: svm.SVC(kernel='linear', C=1, gamma=1),
                    kfold_splits=kfold_splits,
                    return_scores=True,
                    model_callback=model_callback)
	print('\nScore: %.3f +/- %.3f' %(kfold,np.std(scores)))
	print('\nConfusion Matrix:\n', con_matrix)
	f1scores = confusion_matrix_f1_scores(con_matrix)
	for k,f1 in enumerate(f1scores):
		print('f1, Class '+str(k+1)+': %.3f' %(f1))

def Main(file):
	print('\nSupport Vector Classifier:')
	build_and_train_SVM(file)

if __name__=='__main__' and len(sys.argv) > 1:
   import argparse
   directory = sys.argv[1]
   Main(directory)
	


