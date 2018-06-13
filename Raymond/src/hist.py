#!/usr/bin/env python3

from os import sys
from skimage import io, exposure
from skimage.filters import try_all_threshold
from matplotlib import pyplot as plt
import numpy as np

def Main():
   file_names = sys.argv[1:]

   if len(file_names) == 0:
      print('No files to process in program arguments')
      return

   for name in file_names:
      img = io.imread(name, as_gray=True)

      # contrast adjustment
      p2, p98 = np.percentile(img, (2, 98))
      img = exposure.rescale_intensity(img, in_range=(p2, p98))

      #plt.imshow(img, cmap=plt.cm.gray)
      #plt.show()

      fig, ax = try_all_threshold(img, figsize=(10, 8), verbose=False)
      plt.show()

      accept = input('Continue ? (Y/N) ')
      if len(accept) and accept.upper()[0] != 'Y': break

if __name__ == '__main__':
   Main()

