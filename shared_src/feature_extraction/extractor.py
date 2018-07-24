#!/usr/bin/env python3
'''
author : Raymond Sutrisno

This python script takes in as command line argument the name of a directory
which exhibits the following structure to classifiy each image file.

      example:
        -my_directory/
            |----A/
            |    |- image1.jpg
            |    |- image2.jpg
            |
            |----B/
            |    |- image3.jpg
            |
            |----C/
                 |- image4.jpg
                 |- image5.jpg
                 |- image6.jpg

         The images get classified as follows
            class   file_name
               A    image1.jpg
               A    image2.jpg
               B    image3.jpg
               C    image4.jpg
               C    image5.jpg
               C    image6.jpg

This script then saves this data along with the corresponding extracted
features in pandas dataframe. The data is saved as csv file named 'features.csv'

See documentation of the function @extract_features_into_pandas_frame for
more information on the features extracted. 

To run, simpy type as follows in shell:

$ python3 extractor.py [name of directory containing images in corresponding label folders]

'''

import pandas as pd
import numpy as np
from skimage import io as skimio
import cv2 as cv
from matplotlib import pyplot as plt
import scipy.stats as stats
import scipy.io as scio

def local_strided_features(img, feature_extractor, n=2, global_feat=None):
   """
      This function performs feature extractions on localized areas of the image
      using a bounding box which is strided across the image. 

      The box and stride dimensions are calculated as follows
         box_height = floor(image_rows / N)
         box_width  = floor(image_columns / N)
         stride_y   = floor(box_height / 2)
         stride_x   = floor(box_width / 2)

      @parameters:
         @img - 2D numpy array representing the image
         @feature_extractor - function or lambda which computes the feature extraction.
                              must only have as input a 2D numpy array.
         @n - an int used for calculating N
         @global_feat - None, or a real number representing the feature extraction from
                        the image at the global level

      @returns:
         if @global_feat is None:
            returns the local features as a 2D numpy array
         else:
            returns the square root of the sum of squared differences of
            the local features and @global_feat
   """
   box_height = img.shape[0] // n
   box_width  = img.shape[1] // n
   stride_y = box_height // 2
   stride_x = box_width // 2

   features = np.zeros((2*n-1, 2*n-1))
   for i in range(2 * n - 1):
      for j in range(2 * n - 1):
         x = i * stride_x
         y = j * stride_y
         features[i, j] = feature_extractor(img[y:y+box_height, x:x+box_width])

   return np.linalg.norm(features - global_feat) if global_feat is not None else features
### end of def local_strided_features

def entropy(img):
   """
      This function calculates the entropy of an image
      @parameters:
         @img - 2D numpy array of dtype uint8 or some other discretized number form
      @returns:
         the entrop
   """

   freqs = stats.itemfreq(img)[:,1] # extract frequencies of values
   freqs = freqs / (img.shape[0] * img.shape[1]) # normalize frequencies into probabilities
   return stats.entropy(freqs, base=2) # calculate entropy with log base 2
### end of def entropy

def std_dev(img):
   """
      This function calculates the standard deviation of pixel values in an image
      @parameters:
         @img - 2D numpy array representing the image
   """
   return np.std(img) 
### end of def std_dev

def skew(img):
   """
      This function calculates the skewness of pixel values in an image
      @parameters:
         @img - 2D numpy array representing the image
   """
   return stats.skew(img, axis=None)
### end of def skew

def mean_filter_normalize(imgs):
   """
      @parameters:
         @imgs - 3D numpy array of dtype uint8 where the images are stored. each
                 individual image is indexed using the 3rd axis. 
                 imgs[:,:,i] # this indexes the i'th image
   """
   mean_filter = np.mean(imgs, axis=2) # calculate mean image
   mean_bright = np.mean(mean_filter)  # calculate mean brightness of mean image
   ret = np.empty(shape=imgs.shape)    # initialize return array

   for i in range(imgs.shape[2]):
      ret[:,:,i] = imgs[:,:,i] - mean_filter # subtract mean image from image
      ret[:,:,i] += mean_bright - np.mean(ret[:,:,i])
         # arithmetically shift to have same brightness as mean image

   # normalize between 0 and 255
   mini, maxi = np.min(ret), np.max(ret) # calculate min and max values
   ret = ((ret - mini) / (maxi - mini))  # normalize between 0 and 1
   ret = (ret * 255).astype(np.uint8)    # scale to 255

   return ret
### end of def mean_filter_normalize

def extract_features_into_pandas_frame(grayscale_images, saturation_images):
   """
      This function extracts 3 features and their local strided deviations at stride n=2
      from grayscale_images and saturation_images. The 3 features are entropy,
      standard deviation and skewness.

      @parameters:
         @grayscale_images  - 3D numpy array of grayscale images where each image is indexed
                              using axis 3. grayscale_images[:,:,i] indexes the i'th image.
                              Must be of dtype uint8

         @saturation_images - 3D numpy array of saturation images where each image is indexed
                              using axis 3. saturation_images[:,:,i] indexes the i'th image.
                              Must be of dtype uint8
      @returns:
         @df - pandas dataframe containing the 12 features
             'grayscale_ent',
             'grayscale_ent_devN2',
             'grayscale_std',
             'grayscale_std_devN2',
             'grayscale_skew',
             'grayscale_skew_devN2',
             'saturation_ent',
             'saturation_ent_devN2',
             'saturation_std',
             'saturation_std_devN2',
             'saturation_skew',
             'saturation_skew_devN2',
   """
   assert(grayscale_images is not None and saturation_images is not None)
   assert(grayscale_images.shape == saturation_images.shape)

   n_images = grayscale_images.shape[2]

   # features
   grayscale_ent = np.empty(n_images)
   grayscale_ent_devN2 = np.empty(n_images)
   grayscale_std = np.empty(n_images)
   grayscale_std_devN2 = np.empty(n_images)
   grayscale_skew = np.empty(n_images)
   grayscale_skew_devN2 = np.empty(n_images)

   saturation_ent = np.empty(n_images)
   saturation_ent_devN2 = np.empty(n_images)
   saturation_std = np.empty(n_images)
   saturation_std_devN2 = np.empty(n_images)
   saturation_skew = np.empty(n_images)
   saturation_skew_devN2 = np.empty(n_images)

   for i in range(n_images):
      img_gray = grayscale_images[:,:,i]
      img_sat = saturation_images[:,:,i]

      # obtain global features
      grayscale_ent[i] = entropy(img_gray)
      grayscale_std[i] = std_dev(img_gray)
      grayscale_skew[i] = skew(img_gray)
      saturation_ent[i] = entropy(img_sat)
      saturation_std[i] = std_dev(img_sat)
      saturation_skew[i] = skew(img_sat)

      # obtain local striding feature extractions
      grayscale_ent_devN2[i] = local_strided_features(
         img_gray, entropy, global_feat=grayscale_ent[i])
      grayscale_std_devN2[i] = local_strided_features(
         img_gray, std_dev, global_feat=grayscale_std[i])
      grayscale_skew_devN2[i] = local_strided_features(
         img_gray, skew, global_feat=grayscale_skew[i])
      saturation_ent_devN2[i] = local_strided_features(
         img_sat, entropy, global_feat=saturation_ent[i])
      saturation_std_devN2[i] = local_strided_features(
         img_sat, std_dev, global_feat=saturation_std[i])
      saturation_skew_devN2[i] = local_strided_features(
         img_sat, skew, global_feat=saturation_skew[i])


   # create pandas dataframe
   df = pd.DataFrame()
   df['grayscale_ent'] = grayscale_ent
   df['grayscale_ent_devN2'] = grayscale_ent_devN2
   df['grayscale_std'] = grayscale_std
   df['grayscale_std_devN2'] = grayscale_std_devN2
   df['grayscale_skew'] = grayscale_skew
   df['grayscale_skew_devN2'] = grayscale_skew_devN2
   df['saturation_ent'] = saturation_ent
   df['saturation_ent_devN2'] = saturation_ent_devN2
   df['saturation_std'] = saturation_std
   df['saturation_std_devN2'] = saturation_std_devN2
   df['saturation_skew'] = saturation_skew
   df['saturation_skew_devN2'] = saturation_skew_devN2

   return df
### end of def extract_features_into_pandas_frame

def collect_images_and_labels(directory):
   """
      This function collects the images and labels of these images located in
      a directory. The contents of the directory must be organized into folders
      where images under said folder are labeled as that folder.

      example:
        -my_directory/
            |----A/
            |    |- image1.jpg
            |    |- image2.jpg
            |
            |----B/
            |    |- image3.jpg
            |
            |----C/
                 |- image4.jpg
                 |- image5.jpg
                 |- image6.jpg

         The images get classified as follows
            class   file_name
               A    image1.jpg
               A    image2.jpg
               B    image3.jpg
               C    image4.jpg
               C    image5.jpg
               C    image6.jpg

      @returns:
         @labels     - list of strings of corresponding labels to each file in @file_names
         @file_names - list of strings corresponding to the file names of each image

         @grayscale_images  - 3D numpy array of grayscale images where each image is indexed
                              using axis 3. grayscale_images[:,:,i] indexes the i'th image.
                              dtype is uint8, values range from 0 to 255.
                              Images normalized using the mean_filter_normalize function

         @saturation_images - 3D numpy array of saturation images where each image is indexed
                              using axis 3. saturation_images[:,:,i] indexes the i'th image.
                              dtype is uint8, values range from 0 to 255.
                              Images normalized using the mean_filter_normalize function
   """
   labels = []
   file_names = []
   grayscale_images = []
   saturation_images = []

   # collect images and labels
   for label in os.listdir(directory):
      path = '{}/{}'.format(directory, label)
      image_names = os.listdir(path) # names of images in label folder

      # record labels and names
      labels.extend([label]*len(image_names))
      file_names.extend(image_names)

      for file_nm in image_names:
         name = '{}/{}'.format(path, file_nm) # name of file of image
         img = skimio.imread(name) # read image

         img = img[:-128,:,:] # crop scalebar
         img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)      # grayscale
         img_sat = cv.cvtColor(img, cv.COLOR_RGB2HSV)[:,:,1] # saturation

         # collect images
         grayscale_images.append(img_gray)
         saturation_images.append(img_sat)

   # convert to numpy arrays
   grayscale_images = np.moveaxis(np.array(grayscale_images), 0, -1)
   saturation_images = np.moveaxis(np.array(saturation_images), 0, -1)

   # normalize images, this removes artifacts
   grayscale_images = mean_filter_normalize(grayscale_images)
   saturation_images = mean_filter_normalize(saturation_images)

   return labels, file_names, grayscale_images, saturation_images
### end of def collect_images_and_labels

# ============== # Main Area # ============== #

import os
import sys

def Main(directory):
   # collect images and labels from directory
   labels, file_names, grayscale_images, saturation_images = collect_images_and_labels(directory)

   scio.savemat('gray_and_sat.mat', {'grays':grayscale_images, 'sats':saturation_images})

   """
   # extract features into the form of  pandas data frame
   df = extract_features_into_pandas_frame(grayscale_images, saturation_images)

   # add labels and filenames columns
   df['labels'] = labels
   df['file_names'] = file_names

   df.to_pickle('PICKLED_FEATURES')

   # write as csv
   df.to_csv('features.csv')
   """

### end of def Main

if __name__=='__main__' and len(sys.argv) > 1:
   import argparse
   directory = sys.argv[1]
   Main(directory)

