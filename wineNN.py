import numpy as np
import pandas as pd
import pylab as p
import matplotlib.pyplot as plt
from sklearn import cross_validation as cv
from sklearn import metrics as m
from scipy import optimize as op
import neuralNetwork as nn
import sys
sys.path.append('/home/matt/Desktop/DataScience/pybrain')
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
### Random constants and stuff
makePlots = 0
nClasses = 10
IL = 11
HL = 40
OL = 10

### This will hopefully be a method for predicting wine quality
### based on the results of some chemical tests...

white_df = pd.read_csv('winequality-white.csv',sep=';',header=0)
white = white_df.values


#ok, great, we can get our feature matrices, X, and 
#answer vectors, y
nFeatures = np.shape(white)[1]-1
whiteX = white[:,0:nFeatures]
whiteY = white[:,nFeatures]

#Let's try to automatically make some very simple plots
featureNames = ["fixed acidity", "volatile acidity",\
               "citric acid", "residual sugar", "chlorides"\
               , "free sulfur dioxide", "total sulfur dioxide"\
               , "density", "pH", "sulphates", "alcohol"]

if makePlots:
    for i in range(0,nFeatures):
        xstring = featureNames[i]
        ystring = "Quality"
        
        white_xdata = whiteX[:,i]
        plt.figure()
        plt.plot(white_xdata,whiteY,'bx')
        plt.xlabel(xstring)
        plt.ylabel(ystring)
        F = p.gcf()
        xstring = xstring.replace (" ", "_")
        newFilename = "%s.eps" % (xstring)
        F.savefig(newFilename, bbox_inches='tight')
        

### Ok, so we probably want to split our data into training and testing sets. 
### Let's start by trying out a stratified k-fold CV
whiteSKF = cv.StratifiedKFold(whiteY, k=3)

for white_train_index, white_test_index in whiteSKF:
    whiteX_train, whiteX_test = whiteX[white_train_index], whiteX[white_test_index]
    whiteY_train, whiteY_test = whiteY[white_train_index], whiteY[white_test_index]
    

    #let's actually build a yy matrix
    m = np.shape(whiteX_train)[0]
    yy = np.zeros((m,nClasses))
    for i in range(0,m):
        for j in range(0,nClasses):
            if (whiteY_train[i] == j+1):
                yy[i,j] = 1

    X = whiteX_train
    net = buildNetwork(IL,HL,OL)
    ds = SupervisedDataSet(nFeatures,nClasses)
    for i in range(0,m):
        ds.addSample(X[i,:],yy[i,:])
        
    print "successfully loaded %d training examples..."%(m)
    
    trainer = BackpropTrainer(net, ds)
    trainer.trainUntilConvergence()
    
    print "trained neural network until convergence..."
 
    

        


