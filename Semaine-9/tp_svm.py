 import pickle
 import gzip

 import numpy
 # Load the dataset
 with gzip.open("mnist.pkl.gz", "rb") as ifile:
 train, valid, test = pickle.load(ifile, encoding="latin-1")

