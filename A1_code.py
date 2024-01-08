#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 13:48:03 2020

@author: alexisng
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold 

#--------------------FUNCTIONS------------------------#
def matrix(X,M):
    size = len(X)
    dummy = np.ones((size,1)) 
    
    matrix = dummy #dummy column
        
    for i in range(M):
        x_no = X ** (i+1) 
        matrix = np.c_[matrix, x_no] #concatenating columns as M is increasing
    return matrix

#computing W parameters    
def computeW(X,t):
    global w
    A = np.dot(X.T,X)
    A1 = np.linalg.inv(A)
    t1 = np.dot(X.T,t)
    w = np.dot(A1,t1)
    return w

#Training model with regularization 
def W_withRegularization(X,t,value):
    B = np.zeros((10,10)) #creating matrix B with lambdas
    np.fill_diagonal(B,2*value)
    B[0,0] = 0
    
    A = np.dot(X.T,X)
    size = len(X)
    G = A + (size/2 * B)
    A1 = np.linalg.inv(G)
    C = np.dot(X.T,t)
    W = np.dot(A1,C)
    return W

#computing prediction
def fpredict(X,w):
    fpredict = np.dot(X,w)
    return fpredict

#computing error
def error(X,t,f):
    size = len(X)
    diff = np.subtract(t,f)
    error = np.dot(diff,diff.T)/size
    return error

#kfold 
def kfoldFunc(X):
    global averageError
    global w
    
    # split data into training and test sets using kfold
    kf = KFold(n_splits = 5, random_state = 5007, shuffle = True)
    
    avgerror = []

    for train, test in kf.split(X):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train_kf, X_test_kf = X[train], X[test]
        t_train_kf, t_test_kf = t_train[train], t_train[test]

        #Compute linear regression and get test error
        w = computeW(X_train_kf,t_train_kf)
        fpredict_test = fpredict(X_test_kf,w)
        test_error = error(X_test_kf, t_test_kf, fpredict_test)

        print("error: ", test_error)
        avgerror.append(test_error)
    
    averageError = (np.sum(avgerror))/5
    # print("Cross validation error is: " + str(averageError))
    
    return averageError  

#computing final test error after kfold
def testError(train,test):
    global test_error
    computeW(train, t_train)
    fpredict_test = fpredict(test,w)
    test_error = error(test, t_test, fpredict_test)
    
    return test_error

#---PART 1-----#

#sine wave (ftrue)
X_true = np.linspace(0.,1.,100)
f_true = np.sin(4*np.pi*X_true)

#random seed with last 4-digits of student number
np.random.seed(5007)

#Training Sets
X_train = np.linspace(0.,1.,10) #N=10
t_train = np.sin(4*np.pi*X_train)+0.3*np.random.randn(10)

#Validation Sets
X_valid = np.linspace(0.,1.,100) #N = 100
t_valid = np.sin(4*np.pi*X_valid) + 0.3*np.random.randn(100)

#initializing error arrays
error_train = []
error_valid = []

#Computing M=1..10
for M in range(10):
    #Computing predictions on training set
    matrix1 = matrix(X_train, M)
    w = computeW(matrix1,t_train)
    fpredict_train = fpredict(matrix1,w)
    
    #training error
    error_train.append(error(X_train, t_train, fpredict_train))
    
    #Computing predictions on validation set
    matrix2 = matrix(X_valid, M)
    fpredict_valid = fpredict(matrix2,w) #using training coefficients 
   
    #validation error
    error_valid.append(error(X_valid, t_valid, fpredict_valid))
 
   
    #plotting each M
    plt.figure()
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('M = ' + str(M))
    plt.plot(X_train, fpredict_train, 'c-', label ="training prediction")
    plt.plot(X_valid, fpredict_valid, 'm-', label = "validation prediction")
    plt.plot(X_true, f_true, 'k-', label="ftrue")
    plt.plot(X_valid, t_valid, 'bo', label="validation data")
    plt.plot(X_train, t_train, 'ro', label="training data")
    plt.legend(fontsize=7)

print("Training error of each increasing M: " + str(error_train))
print('')
print("Validation error of each increasing M: " + str(error_valid))

#Plotting errors
plt.figure()
plt.xlabel('M')
plt.ylabel('Error')
plt.title('Training and Validation Error')
plt.plot(error_train,'o-', mfc="none", mec="r", ms=5, c="r", label = "Training")
plt.plot(error_valid, 'o-', mfc="none", mec="b", ms=5, c="b", label = "Validation")
plt.legend()

#Plotting average error between valid targets and ftrue
# plt.figure()
# plt.xlabel('x')
# plt.ylabel('t')
# plt.title('Average Error between Validation Targets and f_true')
# avgError = np.subtract(t_valid,f_true)
# avgError = abs(avgError)
# plt.plot(X_true, avgError, label = 'avg error')
# plt.plot(X_true, f_true, 'k-', label="fTrue")
# plt.plot(X_valid, t_valid, 'bo', label="validation data")
# plt.plot(X_train, t_train, 'ro', label="training data")
# plt.legend(fontsize=7)

#compute prediction for overfitting
matrix_M9 = matrix(X_train,9)
W1 = W_withRegularization(matrix_M9,t_train,1e-09) #computing when lambda = 1e-09

fpredict_valid2 = fpredict(matrix2,W1)
lambda_error = error(X_valid, t_valid, fpredict_valid2) #compute validation error
print("")
print("The validation error for lambda1 (eliminating overfitting) is " + str(lambda_error))
plt.figure()
plt.xlabel('x')
plt.ylabel('t')
plt.title('λ1 = 0.1e-09')
plt.plot(X_valid, fpredict_valid2)
plt.plot(X_valid, t_valid, 'bo', label = 'validation data')
plt.plot(X_train, t_train, 'ro', label='training data')
plt.plot(X_true,f_true, 'k', label='ftrue')
plt.legend(fontsize=7)

#compute prediction for underfitting
matrix_M9 = matrix(X_train,9) #M=9
W1 = W_withRegularization(matrix_M9,t_train,1) #Computing when lambda = 1

fpredict_valid2 = fpredict(matrix2,W1)
lambda_error = error(X_valid, t_valid, fpredict_valid2) #compute validation error
print('')
print("The validation error for lambda2 (underfitting) is " + str(lambda_error))

plt.figure()
plt.xlabel('x')
plt.ylabel('t')
plt.title('λ2 = 1')
plt.plot(X_valid, fpredict_valid2)
plt.plot(X_valid, t_valid, 'bo', label = 'validation data')
plt.plot(X_train, t_train, 'ro', label='training data')
plt.plot(X_true,f_true, 'k', label='ftrue')
plt.legend(fontsize=7)



#-------------------------------#

#---PART 2-----#

#Load boston data
# boston = load_boston()
# print(boston.DESCR)

#import data set from scikit
X, t = load_boston(return_X_y=True)
#print(X.shape) #(506,13)
N = len(X)  # number of rows
#print(N) #506

#train/test split 
X_train, X_test, t_train, t_test = train_test_split(X, t, test_size = 1/5, random_state = 5007)
#print(X_train.shape) # X_train is 2D, but y_train is 1D
# print(t_train.shape)
M = len(X_test) #number rows in test set
N = len(X_train) #number rows in train set
#print(N, M) #379 127

# #---------------CROSSVALIDATION AND TEST ERRORS WITHOUT BASIS EXPANSION----------#
totalTestError = [None]
crossValidError = [None]

S = [] #initializing subset S
dummyTrain = np.ones((len(X_train),1)) 
dummyTest= np.ones((len(X_test),1))
matrix = dummyTrain #initializing dummy matrix

for k in range(13):
    
    print("For k = " + str(k+1))
    print('')
    totalavgError = []
    
    for i in range(13):
          
        #to check if there are existing features in S
        if i in S:
            totalavgError.append(100) #adding random value in the final index
            continue
    
        #setting up training data
        X1_train = np.c_[matrix, X_train[:,i]]
        
        #perform kfold
        kfoldFunc(X1_train)
        print("average error of f" + str(i+1) + " is" + str(averageError))
        print('')
        totalavgError.append(averageError)
                
    index_min = np.argmin(totalavgError) #to find index with smallest error    
    print("Smallest cross-validation error is " + str(min(totalavgError)))    
        
    #set up X_train to do test error
    S.append(index_min)
    trainMatrix = np.c_[dummyTrain, X_train[:,S[:len(S)]]]
    testMatrix = np.c_[dummyTest, X_test[:,S[:len(S)]]]
 
    #compute testError
    testError(trainMatrix, testMatrix)
    print("The test error is " + str(test_error))
    
    N = []
    for i in S:
        N.append(i + 1)
    print("S" + str(k+1) + ": " + str(N))
    
    print("W parameters:" + str(w))
    
    print('-------------------------------------------------------------------')
    
    
    matrix = trainMatrix #set up matrix with the chosen features used for next k
      
    #array of all the errors (used to plot)      
    crossValidError.append(min(totalavgError))
    totalTestError.append(test_error)
    
#printing the features in S
N = []
for i in S:
    N.append(i + 1)
print("S" + str(k+1) + ": " + str(N))


# plotting
plt.figure()
plt.xlabel('k')
plt.ylabel('Error')
plt.title('Cross-validation Error and Test Errors without Basis Expansion')
plt.plot(totalTestError,'o-', mfc="none", mec="r", ms=5, c="r", label = "Test")
plt.plot(crossValidError, 'o-', mfc="none", mec="b", ms=5, c="b", label = "Cross-Validation")
plt.legend()

#---------------CROSSVALIDATION AND TEST ERRORS WITH BASIS EXPANSION----------#

#Basis Expasion model: f(x) = ln(x)
def model1(X,S):
    global x1
    size = len(X)
    dummy = np.ones((size,1))
    x = np.c_[dummy, X[:,S[:len(S)]]]
    new_col = np.log(X[:,S[:len(S)]])
    x1 = np.c_[x, new_col]
    return x1


#Basis Expansion model: f(x) = x^2 
def model2(X,S):
    global x2
    size = len(X)
    dummy = np.ones((size,1))
    x = np.c_[dummy, X[:,S[:len(S)]]]
    new_col = X[:,S[:len(S)]]**2
    x2 = np.c_[x, new_col]    
    return x2

#Basis Expansion model: f(x) = sqrt(x) 
def model3(X,S):
    global x3
    size = len(X)
    dummy = np.ones((size,1))
    x = np.c_[dummy, X[:,S[:len(S)]]]
    new_col = np.sqrt(X[:,S[:len(S)]])
    x3 = np.c_[x, new_col]    
    return x3

#Basis Expansion model: f(x) = xi*xm
def model4(X,S):
    global x4
    size = len(X)
    dummy = np.ones((size,1))
    x4 = np.c_[dummy, X[:,S[:len(S)]]]
    product = X[:,S[0]]
    
    for i in range(1, len(S)):
        product = product * X[:,S[i]]
        x4 = np.c_[x4,product]
  
    return x4

#Basis Expansion on S

#for k = 1 to 7
testError_basis = [None]
crossValidError_basis = [None]

for k in range(8):
    
    avgCV = []
    avgT = []

            
    S_temp = S[:k+1]
    # print(S_temp)
    N = []
    for i in S_temp:
        N.append(i + 1)
    print("For S" + str(k+1) + ": " + str(N))
    
    print('')
    
    #model1
    print("Model 1: f(x) = ln(x)")
    X1_train1 = model1(X_train,S_temp)
    X1_test1 = model1(X_test,S_temp)
    
    kfoldFunc(X1_train1)
    testError1 = testError(X1_train1, X1_test1)
    print("Test error is: " + str(testError1)) 
    
    avgCV.append(averageError)
    avgT.append(testError1)

    print('')
    
    #model2
    print("Model 2: f(x) = x^2")
    X1_train2 = model2(X_train,S_temp)
    X1_test2 = model2(X_test,S_temp)   

    kfoldFunc(X1_train2)
    testError2 = testError(X1_train2, X1_test2)
    print("Test error is: " + str(testError2)) 
    
    avgCV.append(averageError)
    avgT.append(testError2)
    
    print('')
    
    #find the smallest errors between all models and append to final array

    crossValidError_basis.append(min(avgCV))
    index_min = np.argmin(avgCV)
    testError_basis.append(avgT[index_min])
    
    print('-------------------------------------------------------------------')

#For k=8 to 12
for k in range(8,12):
    
    avgCV = []
    avgT = []

            
    S_temp = S[:k+1]
    # print(S_temp)
    N = []
    for i in S_temp:
        N.append(i + 1)
    print("For S" + str(k+1) + ": " + str(N))
     
    print('')
    
    #model2
    print("Model 2: f(x) = x^2")
    X1_train2 = model2(X_train,S_temp)
    X1_test2 = model2(X_test,S_temp)   

    kfoldFunc(X1_train2)
    testError2 = testError(X1_train2, X1_test2)
    print("Test error is: " + str(testError2)) 
    
    avgCV.append(averageError)
    avgT.append(testError2)
    
    print('')
    
    #model3
    print("Model 3: f(x) = sqrt(x)")
    X1_train3 = model3(X_train,S_temp)
    X1_test3 = model3(X_test,S_temp)   

    kfoldFunc(X1_train3)
    testError3 = testError(X1_train3, X1_test3)
    print("Test error is: " + str(testError3)) 
    
    avgCV.append(averageError)
    avgT.append(testError3)
    
    #find the smallest errors between all models and append to final array

    crossValidError_basis.append(min(avgCV))
    index_min = np.argmin(avgCV)
    testError_basis.append(avgT[index_min])

    print('-------------------------------------------------------------------')
 
#for k = 13
k = 12
S_temp = S[:k+1]
# print(S_temp)
N = []
for i in S_temp:
    N.append(i + 1)
print("For S" + str(k+1) + ": " + str(N))
 
print('')

#model2
print("Model 2: f(x) = x^2")
X1_train2 = model2(X_train,S_temp)
X1_test2 = model2(X_test,S_temp)   

kfoldFunc(X1_train2)
testError2 = testError(X1_train2, X1_test2)
print("Test error is: " + str(testError2)) 

avgCV.append(averageError)
avgT.append(testError2)

print('')

#model4
print("Model 4: f(x) = xi * xm")
X1_train4 = model4(X_train,S_temp)
X1_test4 = model4(X_test,S_temp)   

kfoldFunc(X1_train4)
testError4 = testError(X1_train4, X1_test4)
print("Test error is: " + str(testError4)) 

avgCV.append(averageError)
avgT.append(testError4)

#find the smallest errors between all models and append to final array

crossValidError_basis.append(min(avgCV))
index_min = np.argmin(avgCV)
testError_basis.append(avgT[index_min])   



#plotting
plt.figure()
plt.xlabel('k')
plt.ylabel('Error')
plt.title('Cross-validation Error and Test Errors with Basis Expansion')
plt.plot(totalTestError,'o-', mfc="none", mec="r", ms=5, c="r", label = "Test Error")
plt.plot(crossValidError, 'o-', mfc="none", mec="b", ms=5, c="b", label = "Cross-Validation Error")
plt.plot(testError_basis, 'o-', mfc="none", mec="k", ms=5, c="k", label = "Test Error w/ Basis Expansion")
plt.plot(crossValidError_basis,'o-', mfc="none", mec="g", ms=5, c="g", label = "Cross-Validation w/ Basis Expansion")
plt.legend()
    
 







