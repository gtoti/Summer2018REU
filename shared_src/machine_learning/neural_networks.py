#!/usr/bin/env python3


import numpy as np
from keras.models import Sequential
from keras.layers import Dense

from utils

def build_keras_sequential(activation='sigmoid', architecture=(12,5)):
   assert(len(architecture) >= 2)
   model = Sequential()
   input_dim = architecture[0]
   for units in architecture[1:]:
      layer = Dense(units, activation=activation, input_dim=input_dim)
      model.add(layer)
      input_dim = units
   return model

def Main():
   model_configurations = [
      (12, 5),
      (12, 8, 5),
      (12, 10, 7, 5),
   ]

   pass

if __name__=='__main__': Main()
