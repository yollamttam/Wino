import numpy as np
import pandas as pd
import pylab as p
import matplotlib.pyplot as plt
from sklearn import cross_validation as cv
from sklearn import metrics as metric
from scipy import optimize as op
import neuralNetwork as nn
import sys
import time
sys.path.append('/home/matt/Desktop/DataScience/pybrain')
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer



def neuralNetwork():
    
    ### Random constants and stuff
    makePlots = 0
    nClasses = 10
    IL = 400
    HL = 25
    OL = 10
    
    ### This will hopefully be a method for predicting wine quality
    ### based on the results of some chemical tests...

    numbers_df = pd.read_csv('HandwrittenNumbers_XandY.txt')
    numbers = numbers_df.values
    

    #ok, great, we can get our feature matrices, X, and 
    #answer vectors, y
    nFeatures = np.shape(numbers)[1]-1
    numbersX = numbers[:,0:nFeatures]
    numbersY = numbers[:,nFeatures]
    

    ### Ok, so we probably want to split our data into training and testing sets. 
    ### Let's start by trying out a stratified k-fold CV
    numbersSKF = cv.StratifiedKFold(numbersY, k=3)

    for numbers_train_index, numbers_test_index in numbersSKF:
        numbersX_train, numbersX_test = numbersX[numbers_train_index], numbersX[numbers_test_index]
        numbersY_train, numbersY_test = numbersY[numbers_train_index], numbersY[numbers_test_index]
        
        abridge = 1
        if abridge:
            numbersX_train = numbersX_train[0:100,:]
            numbersY_train = numbersY_train[0:100]
            
        print "Sizes of training and cv data:"
        print np.shape(numbersX_train)
        print np.shape(numbersX_test)
            
        #let's actually build a yy matrix
        m = np.shape(numbersX_train)[0]
        mval = np.shape(numbersX_test)[0]
            

        yy = np.zeros((m,nClasses))
        yyVal = np.zeros((mval,nClasses))
        for i in range(0,m):
            for j in range(0,nClasses):
                if (numbersY_train[i] == j+1):
                    yy[i,j] = 1
           
        for i in range(0,mval):
            for j in range(0,nClasses):
                if (numbersY_test[i] == j+1):
                    yyVal[i,j] = 1
                
        X = numbersX_train
        Xval = numbersX_test

        net = buildNetwork(IL,HL,OL)

        ds = SupervisedDataSet(nFeatures,nClasses)
        for i in range(0,m):
            ds.addSample(X[i,:],yy[i,:])
        
        print "successfully loaded %d training examples..."%(m)
    
        dsval = SupervisedDataSet(nFeatures,nClasses)
        for i in range(0,mval):
            dsval.addSample(Xval[i,:],yyVal[i,:])
        print "successfully created cv dataset..."


        trainer = BackpropTrainer(net, ds)
        trainer.trainUntilConvergence(verbose = False, continueEpochs = 2)

    
        fullpredict = net.activateOnDataset(dsval)
        print fullpredict 
        
        fullpredictT = np.zeros(np.shape(fullpredict))
        for i in range(0,mval):
            for j in range(0,nClasses):
                if (fullpredict[i,j] == max(fullpredict[i,:])):
                    fullpredictT[i,j] = 1
                
            

        success = metric.precision_score(yyVal,fullpredictT)
        print success

    return yyVal,fullpredictT



