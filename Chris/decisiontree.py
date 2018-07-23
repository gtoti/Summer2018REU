'''
author: Chris Kim

'''

from sklearn.datasets import load_breast_cancer
from id3 import Id3Estimator
#from id3 import export_graphiviz
import pandas as pd

if __name__=='__main__':
	#df = pd.read_csv('features.csv')

	#estimator = Id3Estimator()
	#estimator =
	print(load_breast_cancer().target)