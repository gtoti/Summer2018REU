#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

def pre_process_data(
   file_name, pickled=True, feature_cols=[], label_col=-1, drop=[],
   one_hot=False, shuffle=True, standard_scale=False, index_col=None):

   """
      This function reads in a data file and performs some preprocessing
      depending on passed parameters and returns 2 numpy arrays corresponding
      to selected features and labels as a tuple. feature columns and label columns
      are selected after dropping

      parameters:
         @file_name             - String of file or file buffer

         @pickled=True          - if true, the file is assumed to be in pickled form, else
                                   the file is assumed to be in .csv form

         @feature_cols=[]       - list of strings corresponding to columns or integers
                                   corresponding to columns in which the features are
                                   selected. Order is preserved from the pandas data frame.

                                   By default @feature_cols is an empty array, if this is the
                                   case, every column except the label_column 

         @label_col=-1          - string corresponding to the name of the label column or
                                   integer corresponding to the index of the index of the
                                   label column

         @drop=[]               - list of strings of columns to drop

         @one_hot=False         - whether or not to one hot encode labels,
                                   (only works for classification)

         @shuffle=True          - whether or not to shuffle the data set

         @standard_scale=False  - whether or not to standard scale the features.
                                   Standard scaling of a feature is defined as
                                   subtracting the mean of that feature from every example
                                   and dividing each example by the feature's standard deviation.

         @index_col=None        - only applicable if @pickled=False, for csv mode.
                                   If your csv file has a index column, specify it as an integer
                                   corresponding to the index of the index column

      returns:
         @features  - 2D numpy array where Each example is a row of this numpy array,
                      the columns are the features

         @labels    - Labels corresponding to each example in features

   """

   if pickled:
      df = pd.read_pickle(file_name)
   else:
      df = pd.read_csv(file_name, index_col=index_col)

   if drop:
      df = df.drop(columns=drop)

   print(df.columns)

   if isinstance(label_col, int):
      label_col = df.columns[label_col]

   assert(isinstance(feature_cols, (list, tuple)))

   if not feature_cols:
      feature_cols =[f for f in df.columns if f != label_col]
   elif all(isinstance(f, int) for f in feature_cols):
      feature_cols =[f for i, f in enumerate(df.columns) if i in feature_cols and f != label_col]
   else:
      assert(all(isinstance(f, str) for f in feature_cols))
      feature_cols =[f for f in df.columns if f in feature_cols]

   features = df[feature_cols].values
   labels = df[label_col].values

   if one_hot:
      labels = np.eye(np.unique(labels).shape[0])[labels]

   if shuffle:
      indices = np.random.permutation(len(features))
      features = features[indices]
      labels = labels[indices]

   if standard_scale:
      scaler = StandardScaler()
      features = scaler.fit_transform(features)

   return features, labels

def kFold_Scikit_Model_Trainer(
   features, labels, model_constructor, kfold_splits=10,
   return_scores=False, model_callback=None):
   """
      This function performs kFold cross validation training on a scikit model
      that implements fit and score methods. Returns the mean cross validation
      accuracy and if the parameter @return_scores is true each individual score as
      a list of scores.

      parameters:
         features               - 2D numpy array of features where each row is an example
                                   and the columns of each row are each feature
         labels                 - numpy vector of labels corresponding to each row in features
         model_constructor      - constructor for scikit model
         kfold_splits=10        - number of k folds to perform k fold cross validation
         return_scores=False    - option to return scores or not
         model_callback=None    - function or callable that takes 3 parameters:
                                    @model - the model
                                    @train - the indices used for training
                                    @test  - the indices used for testing

                                    called every kth fold

      returns:
         @kfold_accuracy - the average cross validation accuracy among each k fold
         @scores         - List of scores from each fold. Only returned if
                            parameter @return_scores is true

   """

   kfold = StratifiedKFold(n_splits=kfold_splits, random_state=1).split(features,labels)
   scores = []
   for k, (train,test) in enumerate(kfold):
      model = model_constructor()
      model.fit(features[train], labels[train])
      score = model.score(features[test], labels[test])
      scores.append(score)
      if model_callback is not None and callable(model_callback):
         model_callback(model, train, test)

   kfold_accuracy = np.mean(np.array(scores))

   return (kfold_accuracy, scores) if return_scores else kfold_accuracy

def confusion_matrix_f1_scores(con_matrix):
   """
      Calculates F1 score for each class in a confusion matrix
   """

   # check that con_matrix is a 2D square numpy array
   assert(isinstance(con_matrix, np.ndarray)
          and len(con_matrix.shape) == 2
          and con_matrix.shape[0] == con_matrix.shape[1])

   n_classes = len(con_matrix)
   n_examples = np.sum(con_matrix)

   f1_scores = np.zeros(n_classes)
   for i in range(n_classes):
      tp = con_matrix[i, i]
      fp = np.sum(con_matrix[:, i]) - tp
      fn = np.sum(con_matrix[i, :]) - tp
      #tn = n_examples - tp - fp - fn

      precision = tp / (tp + fp)
      recall = tp / (tp + fn)

      f1_scores[i] = 2. / ((1. / precision) + (1. / recall))

   return f1_scores

if __name__=='__main__':
   print('THIS FILE IS A LIBARY FOR TOTI MACHINE LEARNING, not meant to be run as a script')

