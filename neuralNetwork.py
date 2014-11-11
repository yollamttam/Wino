from math import *
import numpy as np

def sigmoid(z):
    g = 1/(1+np.exp(-1*z))
    return g

def costFunction(Theta1,Theta2,X,y):
    nClasses = 10
    
    #ok, so we want to calculate the cost given the input X, theta matrices
    #Theta1 and Theta2 and known values Y
    
    #first step, we add a column of 1's to the X matrix
    Xrows = np.shape(X)[0]
    Xcolumns = np.shape(X)[1]
    addOnes = np.ones((Xrows,1))
    X = np.hstack((addOnes,X))
    a2 = sigmoid(np.dot(Theta1,np.transpose(X)))
    
    #ok, so now we need to forward propagate again
    #[Theta2] = 10x41 [a2] = (40x12)x(12 x a lot)
    a3 = sigmoid(np.dot(Theta2,a2))

    #ok, so now we need y matrix to be a 10xa lot matrix
    yy = np.zeros((nClasses,Xrows))
    for i in range(0,nClasses):
        for j in range(0,Xrows):
            if (y[j]==i):
                yy[i,j] = i


    return 6


if __name__ == "__main__":
    g = ssigmoid(5)

    
