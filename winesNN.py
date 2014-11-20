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
from pybrain.structure import SigmoidLayer
from REC_curve import *
import pickle


def neuralNetwork():
    
    ### Random constants and stuff
    makePlots = 0
    nClasses = 10
    IL = 11
    HL = 40
    OL = 10
    
    ### This will hopefully be a method for predicting wine quality
    ### based on the results of some chemical tests...

    wines_df = pd.read_csv('train-winequality-white.csv',sep=';',header = 0)
    wines = wines_df.values

    #These may come pre-sorted, which might fuck us up. Let's shuffle them
    np.random.shuffle(wines)

    #ok, great, we can get our feature matrices, X, and 
    #answer vectors, y
    nFeatures = np.shape(wines)[1]-1
    winesX = wines[:,0:nFeatures]
    winesY = wines[:,nFeatures]
    
    for i in range(0,nFeatures):
        mu = np.mean(winesX[:,i])
        sigma = np.sqrt(np.var(winesX[:,i]))
        winesX[:,i] = (winesX[:,i]-mu)/sigma

    ### Ok, so we probably want to split our data into training and testing sets. 
    ### Let's start by trying out a stratified k-fold CV
    winesSKF = cv.StratifiedKFold(winesY, k=3)

    for wines_train_index, wines_test_index in winesSKF:
        winesX_train, winesX_test = winesX[wines_train_index], winesX[wines_test_index]
        winesY_train, winesY_test = winesY[wines_train_index], winesY[wines_test_index]
        
        abridge = 0
        if abridge:
            winesX_train = winesX_train[0:1000,:]
            winesY_train = winesY_train[0:1000]
            
        print "Sizes of training and cv data:"
        print np.shape(winesX_train)
        print np.shape(winesX_test)
            
        #let's actually build a yy matrix
        m = np.shape(winesX_train)[0]
        mval = np.shape(winesX_test)[0]
            
        # this builds a mxn matrix indicating if the mth example belongs 
        # to the nth class
        yy = np.zeros((m,nClasses))
        yyVal = np.zeros((mval,nClasses))
        for i in range(0,m):
            for j in range(0,nClasses):
                if (winesY_train[i] == j+1):
                    yy[i,j] = 1
        
        # and then we do the same for the cross validation set
        for i in range(0,mval):
            for j in range(0,nClasses):
                if (winesY_test[i] == j+1):
                    yyVal[i,j] = 1
                
        X = winesX_train
        Xval = winesX_test

        # set up NN architecture
        net = buildNetwork(IL,HL,OL, outclass=SigmoidLayer)
        ds = SupervisedDataSet(nFeatures,nClasses)
        # add wines data to dataset
        for i in range(0,m):
            ds.addSample(X[i,:],yy[i,:])
        
        print "successfully loaded %d training examples..."%(m)
        
        # add in cross validation data
        dsval = SupervisedDataSet(nFeatures,nClasses)
        for i in range(0,mval):
            dsval.addSample(Xval[i,:],yyVal[i,:])
        print "successfully created cv dataset..."
        
        # run backprop neural network trainer
        trainer = BackpropTrainer(net, ds, learningrate = 0.5)
        trainer.trainUntilConvergence(verbose = False, continueEpochs = 2, validationProportion=0.1)

        # run trained neural network on cross validation set
        fullpredict = net.activateOnDataset(dsval)
        print fullpredict 
        
        # creat mval x n matrix for predictions
        fullpredictT = np.zeros(np.shape(fullpredict))
        for i in range(0,mval):
            for j in range(0,nClasses):
                if (fullpredict[i,j] == max(fullpredict[i,:])):
                    fullpredictT[i,j] = 1
                
        # print success as calculated by metric module
        success = metric.precision_score(yyVal,fullpredictT)
        print success

        
        # build confusion matrix 
        cvvec = np.zeros((mval,1))
        predvec = np.zeros((mval,1))
        for i in range(0,mval):
            for j in range(0,nClasses):
                if (yyVal[i,j] == 1):
                    cvvec[i] = j + 1
                if (fullpredictT[i,j] == 1):
                    predvec[i] = j + 1
        
        confuseMat = metric.confusion_matrix(cvvec,predvec)
        print confuseMat
    return yyVal,fullpredictT


def neuralNetwork_NoCV():
    
    ### Random constants and stuff
    makePlots = 0
    nClasses = 10
    IL = 11
    HL = 40
    OL = 10
    
    ### This will hopefully be a method for predicting wine quality
    ### based on the results of some chemical tests...

    wines_df = pd.read_csv('train-winequality-white.csv',sep=';',header=0)
    wines = wines_df.values
    
    wines_test_df = pd.read_csv('train-winequality-white.csv',sep=';',header=0)
    wines_test = wines_test_df.values

    #These come pre-sorted, which might fuck us up. Let's shuffle them
    np.random.shuffle(wines)

    #ok, great, we can get our feature matrices, X, and 
    #answer vectors, y
    nFeatures = np.shape(wines)[1]-1
    winesX = wines[:,0:nFeatures]
    winesY = wines[:,nFeatures]

    winesX_test = wines_test[:,0:nFeatures]
    winesY_test = wines_test[:,nFeatures]
    
    print "performing feature normalization..."
    for i in range(0,nFeatures):
        mu = np.mean(winesX[:,i])
        sigma = np.sqrt(np.var(winesX[:,i]))
        winesX[:,i] = (winesX[:,i]-mu)/sigma
        winesX_test[:,i] = (winesX_test[:,i]-mu)/sigma

    winesX_train = winesX
    winesY_train = winesY

    print "Sizes of training and cv data:"
    print np.shape(winesX_train)
    print np.shape(winesX_test)
    #let's actually build a yy matrix
    m = np.shape(winesX_train)[0]
    mtest = np.shape(winesX_test)[0]
            
    # this builds a mxn matrix indicating if the mth example belongs 
    # to the nth class
    yy = np.zeros((m,nClasses))
    yytest = np.zeros((mtest,nClasses))
    for i in range(0,m):
        for j in range(0,nClasses):
            if (winesY_train[i] == j+1):
                yy[i,j] = 1
                
                
    for i in range(0,mtest):
        for j in range(0,nClasses):
            if (winesY_test[i] == j+1):
                yytest[i,j] = 1
    
    
                    
    X = winesX_train
    

    # set up NN architecture
    net = buildNetwork(IL,HL,HL,HL,HL,OL, outclass=SigmoidLayer)
    ds = SupervisedDataSet(nFeatures,nClasses)
    dstest = SupervisedDataSet(nFeatures,nClasses)
    # add wines data to dataset
    for i in range(0,m):
        ds.addSample(X[i,:],yy[i,:])
        
    for i in range(0,mtest):
        dstest.addSample(winesX_test[i,:],yytest[i,:])

    print "successfully loaded %d training examples..."%(m)
        
    # run backprop neural network trainer
    trainer = BackpropTrainer(net, ds, learningrate = 0.3)
    trainer.trainUntilConvergence(verbose = False, continueEpochs = 2, validationProportion=0.25)

    # run trained neural network on cross validation set
    fullpredict = net.activateOnDataset(ds)
    fullPredictTest = net.activateOnDataset(dstest)

    # creat mval x n matrix for predictions
    fullpredictT = np.zeros(np.shape(fullpredict))
    fullPredictTtest = np.zeros(np.shape(fullPredictTest))
    for i in range(0,m):
        for j in range(0,nClasses):
            if (fullpredict[i,j] == max(fullpredict[i,:])):
                fullpredictT[i,j] = 1
                
    for i in range(0,mtest):
        for j in range(0,nClasses):
            if (fullPredictTest[i,j] == max(fullPredictTest[i,:])):
                fullPredictTtest[i,j] = 1

    # print success as calculated by metric module
    success = metric.precision_score(yy,fullpredictT)
    successTest = metric.precision_score(yytest,fullPredictTtest)

    print success
    print successTest
        
    # build confusion matrix 
    cvvec = np.zeros((m,1))
    predvec = np.zeros((m,1))
    cvvecTest = np.zeros((mtest,1))
    predvecTest = np.zeros((mtest,1))
    for i in range(0,m):
        for j in range(0,nClasses):
            if (yy[i,j] == 1):
                cvvec[i] = j + 1
            if (fullpredictT[i,j] == 1):
                predvec[i] = j + 1
        
    confuseMat = metric.confusion_matrix(cvvec,predvec)

    print confuseMat

    for i in range(0,mtest):
        incNorm = 0
        for j in range(0,nClasses):
            
            if (yytest[i,j] == 1):
                cvvecTest[i] = j + 1

            if (fullPredictTtest[i,j] == 1):
                predvecTest[i] = j + 1

            

    confuseMat = metric.confusion_matrix(cvvecTest,np.round(predvecTest))
    print confuseMat 

    a = rec_curve(cvvec,predvec)
    a.calc_rec(0.0, 10.0)
    a.display(None)

    a = rec_curve(cvvecTest,predvecTest)
    a.calc_rec(0.0, 10.0)
    a.display(None)

    updateBool = input('Would you like to update your network?')
    if updateBool:
        fileObject = open('winesNNnetwork.pickle','w')
        pickle.dump(net, fileObject)
        fileObject.close()

    return yy,fullpredictT


def winePrediction():
    nClasses = 10

    fileObject = open('winesNNnetwork.pickle','r')
    net = pickle.load(fileObject)

    wines_df = pd.read_csv('train-winequality-white.csv',sep=';',header=0)
    wines = wines_df.values

    wines_test_df = pd.read_csv('test-winequality-white.csv',sep=';',header=0)
    wines_test = wines_test_df.values

    nFeatures = np.shape(wines)[1]-1
    winesX = wines[:,0:nFeatures]
    winesY = wines[:,nFeatures]
    winesX_test = wines_test[:,0:nFeatures]
    winesY_test = wines_test[:,nFeatures]

    print "performing feature normalization..."
    for i in range(0,nFeatures):
        mu = np.mean(winesX[:,i])
        sigma = np.sqrt(np.var(winesX[:,i]))
        winesX[:,i] = (winesX[:,i]-mu)/sigma
        winesX_test[:,i] = (winesX_test[:,i]-mu)/sigma

    winesX_train = winesX
    winesY_train = winesY

    print "Sizes of training and test data:"
    print np.shape(winesX_train)
    print np.shape(winesX_test)
    #let's actually build a yy matrix
    m = np.shape(winesX_train)[0]
    mtest = np.shape(winesX_test)[0]
            
    # this builds a mxn matrix indicating if the mth example belongs 
    # to the nth class
    yy = np.zeros((m,nClasses))
    yytest = np.zeros((mtest,nClasses))
    for i in range(0,m):
        for j in range(0,nClasses):
            if (winesY_train[i] == j+1):
                yy[i,j] = 1
                
                
    for i in range(0,mtest):
        for j in range(0,nClasses):
            if (winesY_test[i] == j+1):
                yytest[i,j] = 1
    
    
                    
    X = winesX_train
    
    ds = SupervisedDataSet(nFeatures,nClasses)
    dstest = SupervisedDataSet(nFeatures,nClasses)
    # add wines data to dataset
    for i in range(0,m):
        ds.addSample(X[i,:],yy[i,:])
        
    for i in range(0,mtest):
        dstest.addSample(winesX_test[i,:],yytest[i,:])

    print "successfully loaded %d training examples..."%(m)
        
    # run trained neural network on cross validation set
    fullpredict = net.activateOnDataset(ds)
    fullPredictTest = net.activateOnDataset(dstest)

    # creat mval x n matrix for predictions
    fullpredictT = np.zeros(np.shape(fullpredict))
    fullPredictTtest = np.zeros(np.shape(fullPredictTest))
    for i in range(0,m):
        for j in range(0,nClasses):
            if (fullpredict[i,j] == max(fullpredict[i,:])):
                fullpredictT[i,j] = 1
                
    for i in range(0,mtest):
        for j in range(0,nClasses):
            if (fullPredictTest[i,j] == max(fullPredictTest[i,:])):
                fullPredictTtest[i,j] = 1

    # print success as calculated by metric module
    success = metric.precision_score(yy,fullpredictT)
    successTest = metric.precision_score(yytest,fullPredictTtest)

    print success
    print successTest
        
    # build confusion matrix 
    cvvec = np.zeros((m,1))
    predvec = np.zeros((m,1))
    cvvecTest = np.zeros((mtest,1))
    predvecTest = np.zeros((mtest,1))
    for i in range(0,m):
        for j in range(0,nClasses):
            if (yy[i,j] == 1):
                cvvec[i] = j + 1
            if (fullpredictT[i,j] == 1):
                predvec[i] = j + 1
        
                



    for i in range(0,mtest):
        incNorm = 0
        for j in range(0,nClasses):
            
            if (yytest[i,j] == 1):
                cvvecTest[i] = j + 1

            if (fullPredictTtest[i,j] == 1):
                predvecTest[i] = j + 1

            

    confuseMat = metric.confusion_matrix(cvvecTest,np.round(predvecTest))
    print confuseMat 

    print np.shape(cvvecTest)
    print np.shape(predvecTest)

    confusion = np.histogram2d(cvvecTest,predvecTest, bins=11,
                               range=[[-0.5,10.5], [-0.5,10.5]])#, weights=None)


    plt.imshow(np.log10(confusion[0]), origin='lower', interpolation='none', vmin=0)
    plt.colorbar()
    plt.title('Confusion Matrix')
    plt.xlabel('true label')
    plt.ylabel('fitted label')
    


    a = rec_curve(cvvecTest,predvecTest)
    a.calc_rec(0.0, 10.0)
    a.display(None)

    

    
