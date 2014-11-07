import numpy as np
import pandas as pd
import pylab as p
import matplotlib.pyplot as plt

### This will hopefully be a method for predicting wine quality
### based on the results of some chemical tests...

white_df = pd.read_csv('winequality-white.csv',sep=';',header=0)
red_df = pd.read_csv('winequality-red.csv',sep=';',header=0)

white = white_df.values
red = red_df.values


#ok, great, we can get our feature matrices, X, and 
#answer vectors, y
nFeatures = np.shape(white)[1]-1
whiteX = white[:,0:nFeatures]
whiteY = white[:,nFeatures]
redX = red[:,0:nFeatures]
redY = red[:,nFeatures]


#Let's try to automatically make some very simple plots
featureNames = ["fixed acidity", "volatile acidity",\
               "citric acid", "residual sugar", "chlorides"\
               , "free sulfur dioxide", "total sulfur dioxide"\
               , "density", "pH", "sulphates", "alcohol"]

for i in range(0,nFeatures):
    xstring = featureNames[i]
    ystring = "Quality"

    white_xdata = whiteX[:,i]
    red_xdata = redX[:,i]
    
    plt.figure()
    plt.plot(white_xdata,whiteY,'bx')
    plt.xlabel(xstring)
    plt.ylabel(ystring)
    F = p.gcf()
    xstring = xstring.replace (" ", "_")
    newFilename = "%s.eps" % (xstring)
    F.savefig(newFilename, bbox_inches='tight')
    
