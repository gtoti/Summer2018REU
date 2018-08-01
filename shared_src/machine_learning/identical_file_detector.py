import os
import sys

def Main(directory):
   files0 = os.listdir(directory[0])
   for i,file in enumerate(files0):
      if file[0] == '_':
         files0[i] = file[1:]
   files1 = os.listdir(directory[1])

   file_set0 = set(files0)
   print(len(file_set0))
   file_set1 = set(files1)
   print(len(file_set1))
   file_match = file_set0.intersection(file_set1)
   for k,file in enumerate(file_match):
      print(str(k+1)+': '+file)
   print('Number of Identical Files:',len(file_match))

if __name__=='__main__' and len(sys.argv) > 1:
   import argparse
   directory = sys.argv[1:]
   Main(directory)