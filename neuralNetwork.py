from math import *
import numpy as np

def sigmoid(z):
    g = 1/(1+np.exp(-1.0*z))
    return g

def sigmoidGradient(z):
    gz = sigmoid(z)
    gradz = gz*(1-gz)
    return gradz

def dummyFunction(X):
    #let's pretend that the only thing that matters is the fixed acidty
    #this should be the first feature
    m = np.shape(X)[0]
    n = np.shape(X)[1]
    Y = X[:,0]
    Y = Y.astype(int)
    
    for i in range(0,m):
        if (Y[i] > 10):
            Y[i] = 10
    
    
    print np.max(Y),np.min(Y)
    return Y


def predict(nn_params,IL,HL,OL,X,nClasses):
    #rebuild Theta matrices
    firstI = HL*(IL+1)
    secondI = OL*(HL+1)
    nn1 = nn_params[0:firstI]
    nn2 = nn_params[firstI::]
    Theta1 = nn1.reshape((HL,IL+1))
    Theta2 = nn2.reshape((OL,HL+1))

    ### forward propagation
    #first step, we add a column of 1's to the X matrix
    Xrows = np.shape(X)[0]
    Xcolumns = np.shape(X)[1]
    addOnes = np.ones((Xrows,1))
    X = np.hstack((addOnes,X))
    a2 = sigmoid(np.dot(Theta1,np.transpose(X)))
    
    addOnes = np.ones((1,np.shape(a2)[1]))
    a2 = np.vstack((addOnes,a2))

    #ok, so now we need to forward propagate again
    #[Theta2] = 10x41 [a2] = (40x12)x(12 x a lot)
    a3 = sigmoid(np.dot(Theta2,a2))

    prediction = np.zeros((np.shape(a3)[1],1))
    #loop over example
    for i in range(0,np.shape(a3)[1]):
        maxVal = 0
        #loop over label
        for j in range(0,np.shape(a3)[0]):
            if (a3[j,i] > maxVal):
                prediction[i] = j+1
                maxVal = a3[j,i]
                
    print np.max(prediction),np.min(prediction)
    return prediction

def costFunction(nn_params,IL,HL,OL,X,y,rLambda):
    firstI = HL*(IL+1)
    secondI = OL*(HL+1)

    nn1 = nn_params[0:firstI]
    nn2 = nn_params[firstI::]
    Theta1 = nn1.reshape((HL,IL+1))
    Theta2 = nn2.reshape((OL,HL+1))


    nClasses = 10
    m = np.size(y)
    m = m*1.0
    #ok, so we want to calculate the cost given the input X, theta matrices
    #Theta1 and Theta2 and known values Y
    
    #first step, we add a column of 1's to the X matrix
    Xrows = np.shape(X)[0]
    Xcolumns = np.shape(X)[1]
    addOnes = np.ones((Xrows,1))
    X = np.hstack((addOnes,X))
    a2 = sigmoid(np.dot(Theta1,np.transpose(X)))
    
    addOnes = np.ones((1,np.shape(a2)[1]))
    a2 = np.vstack((addOnes,a2))

    #ok, so now we need to forward propagate again
    #[Theta2] = 10x41 [a2] = (40x12)x(12 x a lot)
    a3 = sigmoid(np.dot(Theta2,a2))
    
    #ok, so now we need y matrix to be a 10xa lot matrix
    yy = np.zeros((nClasses,Xrows))
    for i in range(0,nClasses):
        for j in range(0,Xrows):
            if (y[j]==i+1):
                yy[i,j] = 1.0
    

                
    #maybe this will help
    ep = 0
    #cost function sans regularization
    Jmatrix = -1.0*(yy*np.log(a3+ep)+(1-yy)*np.log(1-a3+ep))/m
    
    #regularization cost
    Theta1_copy = Theta1
    Theta2_copy = Theta2
    Theta1_copy[:,0] = 0
    Theta2_copy[:,0] = 0

    regCost = rLambda*(np.sum(np.power(Theta1_copy,2))+np.sum(np.power(Theta2_copy,2)))/(2*m)
    J = np.sum(Jmatrix) + regCost
    
    #ok, so delta3 is easy
    delta3 = a3 - yy
    #delta 2, first term
    delta2_firstTerm = np.dot(np.transpose(Theta2),delta3)
    #drop first row
    delta2_firstTerm = delta2_firstTerm[1::,:]
    g2prime = sigmoidGradient(np.dot(Theta1,np.transpose(X)))
    delta2 = delta2_firstTerm*g2prime
    
    Delta1 = np.dot(delta2,X)/m
    Delta2 = np.dot(delta3,np.transpose(a2))/m

    Theta1_copy = Theta1
    Theta2_copy = Theta2
    Theta1_copy[:,0] = 0
    Theta2_copy[:,0] = 0

    Theta1_grad = Delta1 + rLambda*Theta1_copy/m
    Theta2_grad = Delta2 + rLambda*Theta2_copy/m

    grads = np.hstack((Theta1_grad.ravel(),Theta2_grad.ravel()))
    
    return J,grads


if __name__ == "__main__":
    g = ssigmoid(5)

    
