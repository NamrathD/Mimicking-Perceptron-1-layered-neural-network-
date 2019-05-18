# A perceptron is a neural networks with no hidden layers. And as its activation function is a sigmoid function,
# it is quite similar to logistic regression. Therefore, mimicking a perceptron is pretty much similar to logistic as well
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures # for polynomial features
import math

class Perceptron(): # should have functions fit, predict, scores
    para = [] # same as coef_ except contains intercpt values
    coef_ = [] # prints coefficients without intercpt
    intercept = 0
    def __init__(self,max_iter=1000,alpha=0.444):
        self.iter = max_iter
        self.alpha = alpha
    def _intiPara(self,x):
        para = [0]*len(x[0])
        para = np.array(para,dtype=float) # important to define it as float para array
        para = para.reshape(-1,1)
        Perceptron.para  = para
    def _hypothesis(self,x,para):
        z = np.dot(x,para)
        g = (1 / (1 + np.exp(-z)))
        return g
    def _Update_Coef_intercept(self,para):
        Perceptron.coef_ = para[1:,0]
        Perceptron.coef_ = np.array(Perceptron.coef_,dtype=float)
        Perceptron.coef_ = Perceptron.coef_.reshape(1,-1)
        Perceptron.intercept_ = para[0,0]
    def _GD(self,x,y,alpha):
        para = Perceptron.para
        iter = self.iter
        tempPara = para
        m = len(y)

        for i in range(iter):
            hypo = Perceptron._hypothesis(self,x,para)
            for j in range(len(x[0])):
                X_ji = x[:, j]
                X_ji = X_ji.reshape(-1,1)
                tempPara[j] = para[j] - ((self.alpha/m)*(np.sum((hypo-y)*X_ji)))
            para = tempPara
        Perceptron.para = para
        Perceptron._Update_Coef_intercept(self,para)

    def fit(self,x,y): #arguments include X_train and y_train
        if isinstance(x,np.ndarray) == False or isinstance(y,np.ndarray) == False :
            raise TypeError ("Make sure to send numpy array for x and y")
        if len(x.shape) <= 1 or  len(y.shape) <= 1:
            exit("Reshape x and y to get proper dimensions")
        if x.shape[0]!=y.shape[0]:
            exit("make sure to send same number of rows (or samples) for x and y")
        x = np.insert(x,0,1,axis=1)
        y = y.reshape(-1,1)
        Perceptron._intiPara(self,x)
        alpha = 0.01
        Perceptron._GD(self,x,y,alpha)
    def predict_proba(self,x):
        if isinstance(x,np.ndarray) == False :
            raise TypeError ("Make sure to send numpy array for x")
        X_without_intercept = np.array(x)
        x = np.insert(X_without_intercept, 0, 1, axis=1)
        para = Perceptron.para
        y_pred = Perceptron._hypothesis(self,x,para)
        return y_pred
    def predict(self,x,threshold=0.5): #arguments just X_test
        if isinstance(x,np.ndarray) == False :
            raise TypeError ("Make sure to send numpy array for x")
        y_pred = self.predict_proba(x)
        return y_pred >= threshold
    def score(self,x,y): #arguments include X_train and y_train
        if isinstance(x,np.ndarray) == False or isinstance(y,np.ndarray) == False :
            raise TypeError ("Make sure to send numpy array for x and y")
        y_pred = Perceptron.predict(self,x)
        y_pred = np.array(y_pred)
        y_pred = y_pred.reshape(-1,1)
        y= y.reshape(-1,1)

        matched = y_pred==y
        score = (np.sum(matched)/len(y))*100
        return score

Microchips = pd.read_csv("ex2data2.txt")

X = Microchips.drop('Result',axis=1)
y = Microchips['Result']
X = np.array(X)
y = np.array(y)
y = y.reshape(-1,1)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=5)

poly = PolynomialFeatures(degree=2)
print("Printing first row of X before Polynomial Features have been applied:",X_train[0])
X_poly = poly.fit_transform(X_train)
print("Printing first row of X after Polynomial Features (w/ deg =2) have been applied:",X_poly[0])
poly.fit(X_poly,y_train)

X_test_poly = poly.fit_transform(X_test)
poly.fit(X_test_poly,y_test)

clf = Perceptron()
clf.fit(X_poly,y_train)


y_test_pred = clf.predict(X_test_poly)
finalscore = clf.score(X_test_poly,y_test)
coef = clf.coef_
intercept = clf.intercept_

print("Final Parameters values:", coef)
print("Intercept:", intercept)
print("Final Score:",finalscore)