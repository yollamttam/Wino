import numpy as np
import pandas as pd
import pylab as p
import matplotlib.pyplot as plt
from sklearn import cross_validation as cv
from sklearn import metrics as m
from scipy import optimize as op
import neuralNetwork as nn

### Random constants and stuff
makePlots = 0
epsilon = .001
Ninputs = 11
IL = Ninputs
Noutputs = 10
OL = Noutputs
Nhidden = 40
HL = Nhidden
rLambda = 1
nClasses = 10

testing = 1

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

if makePlots:
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
        

### Ok, so we probably want to split our data into training and testing sets. 
### Let's start by trying out a stratified k-fold CV
whiteSKF = cv.StratifiedKFold(whiteY, k=3)
redSKF = cv.StratifiedKFold(redY, k=3)

for white_train_index, white_test_index in whiteSKF:
    whiteX_train, whiteX_test = whiteX[white_train_index], whiteX[white_test_index]
    whiteY_train, whiteY_test = whiteY[white_train_index], whiteY[white_test_index]
    
    #ok, great, now we are ready to choose a ML algorithm...
    #Since we're doing a little drinking, let's try a neural network. 
    #What could go horribly wrong?

    #Ok, so first, we have to decide on a neural network architecture 
    #---> 1 input layer, 1 hidden layer, 1 output layer
    #---> 11 inputs, 40 inputs, 10 outputs



    if (testing):
        whiteY_train = nn.dummyFunction(whiteX_train)
        whiteY_test = nn.dummyFunction(whiteX_test)


    #Theta1 should be 40x12 and randomly initialized.
    Theta1 = 2*epsilon*np.random.random((Nhidden,Ninputs+1))-epsilon
    #Theta2 should be  10x41 and randomly initialized
    Theta2 = 2*epsilon*np.random.random((Noutputs,Nhidden+1))
    initial_nn_params = np.hstack((Theta1.ravel(),Theta2.ravel()))
    nn_params, cost, _, _, _  = op.fmin_cg(lambda t: nn.costFunction(t, IL, HL, OL, whiteX_train, whiteY_train, rLambda)[0], initial_nn_params, fprime = lambda t: nn.costFunction(t,IL,HL,OL,whiteX_train,whiteY_train,rLambda)[1],gtol = 0.001, maxiter = 80, full_output=1)
	
    print nn_params

    #Reshape things to get Theta1,Theta2 back
    firstI = HL*(IL+1)
    nn1 = nn_params[0:firstI]
    nn2 = nn_params[firstI::]
    Theta1 = nn1.reshape((HL,IL+1))
    Theta2 = nn2.reshape((OL,HL+1))

    predict_params = np.hstack((Theta1.ravel(),Theta2.ravel()))
    Ypredict = nn.predict(predict_params,IL,HL,OL,whiteX_test,nClasses)
    success = m.precision_score(whiteY_test,Ypredict)
    print success
    


bothWines = 0
if bothWines:
    
    for red_train_index, red_test_index in redSKF:
        redX_train, redX_test = redX[red_train_index], redX[red_test_index]
        redY_train, redY_test = redY[red_train_index], redY[red_test_index]
        

        


