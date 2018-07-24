#!/usr/bin/env python3

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from utils import pre_process_data

def train_random_forest(features, labels):
   clf = RandomForestClassifier()
   clf.fit(features, labels)
   return clf

def KFold(features, labels):
   yield 

def Main():
   import argparse
   parser = argparse.ArgumentParser()

   file_name = None
   training_data = pre_preprocess_data(file_name, label_col=-2)


   pass

if __name__=='__main__': Main()

