#!/usr/bin/env python3
'''
authors: Chris Kim and Raymond Sutrisno

Optimal model selected using a gridsearch: Architecture @ (12,8,5,4)
										   Regularization @ L1
										   Penalty @ 0.0001
										   Epoch @ 700

Cross-Validation Performance: 0.854 +/- 0.139
Testing Performance: 0.857
Confusion Matrix:
 [[0. 1. 0. 0.]
 [0. 1. 0. 0.]
 [0. 0. 2. 0.]
 [0. 0. 0. 3.]]
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
from ast import literal_eval

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

def get_training_metrics(records, epoch):
	scores = []
	for record in records:
		acc = record['acc']
		scores.append(acc[epoch-1])
	return np.mean(scores), np.std(scores)

def get_validation_metrics(records, epoch):
	scores = []
	for record in records:
		val_acc = record['val_acc']
		scores.append(val_acc[epoch-1])
	return np.mean(scores), np.std(scores)

def train_keras(estimator, data, labels, epoch_num, n_splits=10):
	con_matrix_labels = sorted(np.unique(labels))
	class_num = len(con_matrix_labels)
	con_matrix = np.zeros(shape=(class_num,class_num))
	
	kfold = StratifiedKFold(n_splits=n_splits,random_state=1).split(data,labels)
	records = []

	for j,(train,test) in enumerate(kfold):
		model = estimator
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

	return records, f1scores

def train_keras_sequential(data,labels,config,epoch_num, reg, reg_grp, penalty, metric_count, e_list):
	estimator = build_keras_sequential(architecture=config, reg=reg)
	records, f1scores = train_keras(estimator, data, labels, epoch_num)
	class_num = len(np.unique(labels))

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

	return all_metrics # [score, std, config, regs, epoch, f1mean, class1f1, class2f2, ... , score, std, config, ... ]

def gridsearch(file, metric_num, data, labels, first_layer, second_layer, epoch_col, reg_list):
	metric_count = metric_num

	input_num = data.shape[1]
	output_num = np.unique(labels).shape[0]

	first_layer_nodes = first_layer
	second_layer_nodes = second_layer

	epoch_list = epoch_col
	epoch = 800
	
	model_configurations = []
	for i in first_layer_nodes:
		for j in second_layer_nodes:
			model_configurations.append((input_num, i, j, output_num))
	
	regs = reg_list

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
				dataset[model_count] = train_keras_sequential(data, labels, config, epoch, reg, k, penalty, metric_count, epoch_list)
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

def test_optimal_model(data_train, labels_train, data_test, labels_test, config, reg, epoch):
	n_splits = 10
	con_matrix_labels = sorted(np.unique(labels_train))
	class_num = len(con_matrix_labels)
	con_matrix = np.zeros(shape=(class_num,class_num))

	model = build_keras_sequential(architecture=config, reg=reg)
	model.fit(data_train,
			  to_categorical(labels_train,class_num),
			  epochs=epoch,
			  batch_size=10,
			  verbose=0)
	score = model.evaluate(data_test,to_categorical(labels_test,class_num),batch_size=10,verbose=0) # score = [test loss, test accuracy]
	predictions = model.predict_classes(data_test)
	con_matrix = con_matrix + confusion_matrix(labels_test,predictions,labels=con_matrix_labels)

	#f1scores is 1D np.array with the f1 scores for each classification
	f1scores = confusion_matrix_f1_scores(con_matrix)

	# Drops any NaN f1 scores to compute the mean f1score
	f1scores_noNaN = [x for x in f1scores if str(x) != 'nan']

	print('---------------------------------------------------------------------------')
	print('Testing Neural Network with Optimal Hyperparameters (Architecture @',str(config),', Epochs @',epoch)
	print('___________________________________________________________________________')
	print('Testing Accuracy:',score[1])
	print('Confusion Matrix:\n',con_matrix)
	print('---------------------------------------------------------------------------')

	return score[1], con_matrix

def plot_learning_curve(title, data, labels, config, reg, epoch, ylim=None, train_sizes=np.linspace(.1, 1.0, 5)):
	"""
	Parameters
		----------
		ylim : tuple, shape (ymin, ymax), optional
		    Defines minimum and maximum yvalues plotted.

		cv : int, cross-validation generator or an iterable, optional
		    Determines the cross-validation splitting strategy.
		    Possible inputs for cv are:
		      - None, to use the default 3-fold cross-validation,
		      - integer, to specify the number of folds.
		      - An object to be used as a cross-validation generator.
		      - An iterable yielding train/test splits.

		    For integer/None inputs, if ``y`` is binary or multiclass,
		    :class:`StratifiedKFold` used. If the estimator is not a classifier
		    or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

		n_jobs : integer, optional
		    Number of jobs to run in parallel (default 1).
	"""
	plt.figure()
	plt.title(title)
	if ylim is not None:
	    plt.ylim(*ylim)
	plt.xlabel("Training examples")
	plt.ylabel("Score")

	points_num = train_sizes.shape[0]
	train_scores_mean = np.zeros(points_num)
	train_scores_std = np.zeros(points_num)
	test_scores_mean = np.zeros(points_num)
	test_scores_std = np.zeros(points_num)

	for k, size in enumerate(train_sizes):
		print('-------------------------------------------------')
		print('\nPlotting Training Percentage',size)
		estimator = build_keras_sequential(architecture=config, reg=reg)
		sample_num = int(size * data.shape[0])
		select = np.random.choice(data.shape[0], sample_num, replace=False)
		data_subset = data[select, :]
		labels_subset = labels[select]
		records, f1scores = train_keras(estimator, data_subset, labels_subset, epoch)
		train_scores_mean[k], train_scores_std[k] = get_training_metrics(records, epoch)
		test_scores_mean[k], test_scores_std[k] = get_validation_metrics(records, epoch)
		print('-------------------------------------------------')

	plt.grid()

	plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
					 train_scores_mean + train_scores_std, alpha=0.1, color="r")
	plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
					 test_scores_mean + test_scores_std, alpha=0.1, color="g")
	plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
	plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

	plt.legend(loc='lower right')
	# plt.show()
	return plt

def Main(file):
	metric_count = 6
	
	first_layer_nodes = [10,8,5]
	second_layer_nodes = [8,5]
	epoch_list = [500,600,700,800]
	# kernel_regularizer function applied to kernel weights matrix; activity_regularizer function applies to output of layer
	regs = [[None],
			[regularizers.l1(0.0001), regularizers.l1(0.001)],
			[regularizers.l2(0.0001), regularizers.l2(0.001)]]

	data, labels = pre_process_data(file, pickled=False, feature_cols=[], label_col=-1, drop=['file_names'],
	                                   one_hot=False, shuffle=True, standard_scale=True, index_col=0)
	i = len(labels) * 9//10
	data_train = data[:i]
	labels_train = labels[:i]
	data_test = data[i:]
	labels_test = labels[i:]
	#gridsearch(file,metric_count, data_train, labels_train, first_layer_nodes, second_layer_nodes, epoch_list, regs)
	df = pd.read_excel(file[9:-4]+'_gridsearch[archived].xlsx')
	df = df.sort_values(by=['Mean'],ascending=False)

	output = []
	configs = []
	regs = []
	epochs = []
	for m_num in range(5):
		plot_config = literal_eval(df.iloc[m_num]['Config'])
		configs.append(str(plot_config))
		reg = df.iloc[m_num]['Reg']
		regs.append(reg)
		if reg[-2] == '-':
			penalty = 10**int(reg[-2:])
			if reg[:2] == 'l1':
				plot_reg = regularizers.l1(penalty)
			else:
				plot_reg = regularizers.l2(penalty)
		else:
			plot_reg = None
		plot_epoch = int(df.iloc[m_num]['Epoch'])
		epochs.append(plot_epoch)


		title = "Learning Curves (Neural Networks)"
		print('PLOTTING Model',m_num+1,'...')
		train_sizes=np.linspace(.25, 1.0, 7)
		plot = plot_learning_curve(title, data_train, labels_train, plot_config, plot_reg, plot_epoch, train_sizes=train_sizes)
		
		print('TESTING Model',m_num+1,'...')
		acc, con_matrix =test_optimal_model(data_train, labels_train, data_test, labels_test, plot_config, plot_reg, plot_epoch)
		output.append([acc,con_matrix,plot])

	scores_sorted = sorted(np.array([model[0] for model in output]))
	for k,result in enumerate(output):
		print('\nModel', k+1)
		print('\tSorted',len(scores_sorted)-scores_sorted.index(result[0]), 'Best')
		print('\tConfiguration:',configs[k])
		print('\tRegularization:',regs[k])
		print('\tNo. of Epochs:',epochs[k])
		print('\tAccuracy:',result[0])
		print('\tConfusion Marix:')
		for row in result[1]:
			print('\t\t',row)
		f1scores = confusion_matrix_f1_scores(result[1])
		for i,f1 in enumerate(f1scores):
			print('\tf1, Class str(i+1):',f1)
		result[2].show()

if __name__=='__main__' and len(sys.argv) > 1:
	import argparse
	directory = sys.argv[1]
	Main(directory)