from sklearn import svm
from utils import *


def build_SVM(file):
	features, labels = pre_process_data(file, pickled=False, feature_cols=[], label_col=-1, drop=['file_names'], one_hot=False, shuffle=True):


def Main(file)
	build_SVM(file)

if __name__=='__main__' and len(sys.argv) > 1:
   import argparse
   directory = sys.argv[1]
   Main(directory)
	

# Create SVM classification object 
model = svm.svc(kernel='linear', c=1, gamma=1) 
# there is various option associated with it, like changing kernel, gamma and C value. Will discuss more # about it in next section.Train the model using the training sets and check score
model.fit(X, y)
model.score(X, y)
#Predict Output
predicted= model.predict(x_test)