#!/usr/bin/env python3
'''
authors: Chris Kim and Raymond Sutrisno
'''

import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import StratifiedKFold # class proportions are preserved in each folder
from utils import *

def build_keras_sequential(activation='sigmoid', architecture=(12,5), reg=None):
	model = Sequential()
	input_dim = architecture[0]
	# Adds hidden layers to Keras neural network
	for units in architecture[1:]:
		layer = Dense(units, 
					  activation=activation,
					  input_dim=input_dim,
					  kernel_regularizer=reg)
		model.add(layer)
		input_dim = units
	# Compiles Keras neural network
	model.compile(optimizer='adam',
				  loss='categorical_crossentropy',
				  metrics=['accuracy'])
	return model

def get_validation_metrics(records, epoch):
	scores = []
	for record in records:
		val_acc = record['val_acc']
		scores.append(val_acc[epoch-1])
	return np.mean(scores), np.std(scores)


def train_keras_sequential(data,labels,config,epoch_num, reg, reg_grp, penalty, metric_count, e_list):
	n_splits = 10
	
	con_matrix_labels = sorted(np.unique(labels))
	class_num = len(con_matrix_labels)
	con_matrix = np.zeros(shape=(class_num,class_num))
	
	kfold = StratifiedKFold(n_splits=n_splits,random_state=1).split(data,labels)
	records = []

	for j,(train,test) in enumerate(kfold):
		model = build_keras_sequential(architecture=config, reg=reg)
		print('\tTraining Fold',j+1)
		# Record is a History object: record.history['acc'] and record.history['val_acc']
		record = model.fit(data[train],
						   to_categorical(labels[train],class_num),
						   validation_data=(data[test],to_categorical(labels[test],class_num)),
						   epochs=epoch_num,
						   batch_size=10,
						   verbose=0)
		records.append(record.history)
		predictions = model.predict_classes(data[test])
		con_matrix = con_matrix + confusion_matrix(labels[test],predictions,labels=con_matrix_labels)

	#f1scores is 1D np.array with the f1 scores for each classification
	f1scores = confusion_matrix_f1_scores(con_matrix)

	# Drops any NaN f1 scores to compute the mean f1score
	f1scores_noNaN = [x for x in f1scores if str(x) != 'nan']

	epoch_list = e_list
	all_metrics = [None]*(metric_count+class_num)
	all_metrics = np.array(all_metrics * len(epoch_list))


	reg_grp_dict = {0:'na', 1:'l1', 2:'l2'}
	penalty_dict = {-1:'',0:'-4',1:'-3'}

	for i, epoch in enumerate(epoch_list):
		score, std = get_validation_metrics(records,epoch)
		metrics = [score, std, str(config), reg_grp_dict[reg_grp]+penalty_dict[penalty], epoch, np.mean(f1scores_noNaN)]
		metrics.extend(f1scores)
		for k, metric in enumerate(metrics):
			all_metrics[(i*len(metrics))+k] = metric

	return all_metrics

def Main(file):
	metric_count = 6
	data, labels = pre_process_data(file, pickled=False, feature_cols=[], label_col=-1, drop=['file_names'],
                                   one_hot=False, shuffle=True, standard_scale=True, index_col=0)
	i = len(labels) * 9//10
	data_train = data[:i]
	data_test = data[i:]
	labels_train = labels[:i]
	labels_test = labels[i:]

	input_num = data.shape[1]
	output_num = np.unique(labels).shape[0]

	first_layer_nodes = [10,8,5]
	second_layer_nodes = [8,5]

	epoch_list = [500,600,700,800]
	epoch = 800
	
	model_configurations = []
	for i in first_layer_nodes:
		for j in second_layer_nodes:
			model_configurations.append((input_num, i, j, output_num))
	
	# kernel_regularizer function applied to kernel weights matrix; activity_regularizer function applies to output of layer
	regs = [[None],
			[regularizers.l1(0.0001), regularizers.l1(0.001)],
			[regularizers.l2(0.0001), regularizers.l2(0.001)]]

	# Instantiate the dataset np.array that will contain all the metrics and parameters
	epoch_num = len(epoch_list)
	dataset = [None]*((metric_count+output_num) * epoch_num)
	total = len(model_configurations) * (len(regs[0])+len(regs[1])+len(regs[2]))
	dataset = np.array([[dataset] * total])
	dataset = dataset.reshape(total, (metric_count+output_num)*epoch_num)

	model_count = 0
	penalty = -1
	
	for config in model_configurations:
		for k, reg_group in enumerate(regs):
			for j, reg in enumerate(reg_group):
				print('\nTraining and Cross-Validating Model',model_count+1,'of',total,'...')
				if len(reg_group) > 1:
					penalty = j
				else: # Weights without regularization will not have a penalty
					penalty = -1 
				dataset[model_count] = train_keras_sequential(data_train, labels_train, config, epoch, reg, k, penalty, metric_count, epoch_list)
				model_count += 1

	dataset = dataset.reshape(total*epoch_num, metric_count+output_num)
	
	# Creates a dataframe of the np.array containing all the metrics and parameters
	df = pd.DataFrame(dataset)
	df = df.rename({0:'Mean',1:'Std',2:'Config',3:'Reg', 4:'Epoch',5:'Mean_f1'}, axis = 'columns')
	f1_col_names = {}
	for i in range(output_num):
		f1_col_names.update({i+metric_count:'f1_class'+str(i)})
	df = df.rename(f1_col_names, axis='columns')
	df = df.sort_values(by=['Mean'], ascending = False)

	# Outputs dataframe to an Excel file
	title = file[file.find('_')+1:file.find('.')]
	writer = pd.ExcelWriter(title+'_gridsearch.xlsx') # file name
	df.to_excel(writer,title)
	writer.close()

if __name__=='__main__' and len(sys.argv) > 1:
	import argparse
	directory = sys.argv[1]
	Main(directory)