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
# def build_keras_sequential(activation='sigmoid', architecture=(12,5)):
	model = Sequential()
	input_dim = architecture[0]
	for units in architecture[1:]:
		layer = Dense(units, 
					  activation=activation,
					  input_dim=input_dim,
					  kernel_regularizer=reg)
		'''
		layer = Dense(units, activation=activation, input_dim=input_dim)
		'''
		model.add(layer)
		input_dim = units
	model.compile(optimizer='adam',
				  loss='categorical_crossentropy',
				  metrics=['accuracy'])
	return model

def get_validation_metrics(records, epoch):
	scores = []
	for record in records: # for each record from the 10 folds
		val_acc = record['val_acc']
		scores.append(val_acc[epoch-1])
	return np.array([[np.mean(scores), np.std(scores)]])


def train_keras_sequential(data,labels,config,epoch_num, reg, metric_count):
# def train_keras_sequential(data,labels,config,epoch_num, metric_count):
	n_splits = 2
	
	con_matrix_labels = sorted(np.unique(labels))
	class_num = len(con_matrix_labels)
	con_matrix = np.zeros(shape=(class_num,class_num))
	
	kfold = StratifiedKFold(n_splits=n_splits,random_state=1).split(data,labels)
	records = []

	for j,(train,test) in enumerate(kfold):
		model = build_keras_sequential(architecture=config, reg=reg)
		print('\nTraining Fold',j+1,'|||||||||||||||||||||||||||||||||||||||||||||||||||||||')
		# Record is a History object: record.history['acc'] and record.history['val_acc']
		record = model.fit(data[train],
						   to_categorical(labels[train],class_num),
						   validation_data=(data[test],to_categorical(labels[test],class_num)),
						   epochs=epoch_num,
						   batch_size=10,
						   verbose=1)

		records.append(record.history)
		predictions = model.predict_classes(data[test])
		con_matrix = con_matrix + confusion_matrix(labels[test],predictions,labels=con_matrix_labels)

	f1scores = np.array([confusion_matrix_f1_scores(con_matrix)])
	f1scores_noNaN = [x for x in f1scores if str(x) != 'nan']

	all_metrics = np.zeros((1,f1scores.shape[1]+metric_count))

	for epoch in [500,600,700,800]:
		metrics = np.array([[str(config), epoch, np.mean(f1scores_noNaN)]])
		metrics = np.concatenate((get_validation_metrics(records,epoch), metrics, f1scores), axis=1) # metrics = [acc_mean, acc_std, config, epoch, mean_f1, class1_f1, class2_f1, class3_f1,...]
		all_metrics = np.concatenate((all_metrics, metrics), axis=0)

	return all_metrics[1:,:]

def Main(file):
	metric_count = 5
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

	epoch = 800

	'''
	model_configurations = []
    for i in first_layer_nodes:
		for j in second_layer_nodes:
		model_configurations.append((input_num, i, j, output_num))
	'''
	
	# kernel_regularizer function applied to kernel weights matrix; activity_regularizer function applies to output of layer
	# regs = [regularizers.l2(0.01)]
	# regs = [None]
	regs = []

	dataset = np.zeros((1,output_num+metric_count))

	model_configurations = [(input_num,10,8,output_num)]

	for k, config in enumerate(model_configurations):
		print('\nTraining and Cross-Validating Model',k+1,'of',len(model_configurations),'...')
		metrics = train_keras_sequential(data_train, labels_train, config, epoch, regularizers.l2(0.01), metric_count)
		# metrics = train_keras_sequential(data_train, labels_train, config, epoch, reg, metric_count)
		dataset = np.concatenate((dataset, metrics), axis=0)	
	'''
	for k, config in enumerate(model_configurations):
		for j, reg in enumerate(regs):
			print('\nTraining and Cross-Validating Model',k+j+1,'of',len(model_configurations),'...')
			metrics = train_keras_sequential(data_train, labels_train, config, epoch, reg, metric_count)
			dataset = np.concatenate((dataset, metrics), axis=0)
	'''

	df = pd.DataFrame(dataset[1:,:])
	df = df.rename({0:'Mean',1:'Std',2:'Config',3:'Epoch',4:'Mean_f1'}, axis = 'columns')
	f1_col_names = {}
	for i in range(output_num):
		f1_col_names.update({i+metric_count:'f1_class'+str(i)})
	df = df.rename(f1_col_names, axis='columns')

	print(df)

	title = file[file.find('_')+1:file.find('.')]
	writer = pd.ExcelWriter(title+'_gridsearch.xlsx') # file name
	df.to_excel(writer,title)
	writer.close()

if __name__=='__main__' and len(sys.argv) > 1:

	import argparse
	directory = sys.argv[1]
	Main(directory)