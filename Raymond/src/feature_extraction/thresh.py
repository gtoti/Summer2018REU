#!/usr/bin/env python3

from os import sys
from skimage import io, exposure
from skimage.filters import try_all_threshold
from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv

def Main():
   file_names = sys.argv[1:]

   if len(file_names) == 0:
      print('No files to process in program arguments')
      return

   for name in file_names:
      img = io.imread(name, as_gray=True)
      print('image variance=', (img / 255).var())

      # contrast adjustment
      #p2, p98 = np.percentile(img, (5, 95))
      #img = exposure.rescale_intensity(img, in_range=(p2, p98))

      img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 21, 4)
      #img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 21, 4)

      plt.imshow(img, cmap=plt.cm.gray)
      plt.show()

      #fig, ax = try_all_threshold(img, figsize=(10, 8), verbose=False)
      #plt.show()

      accept = input('Continue ? (Y/N) ')
      if len(accept) and accept.upper()[0] != 'Y': break

if __name__ == '__main__':
   Main()

