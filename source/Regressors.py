import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Linear Regression
class LinearRegression():

    def __init__(self,X,y,alpha = 0.03, iterations = 1500):
        self.nvar = len(X[0])
        self.ndata = len(X)
        self.iterations = iterations
        self.alpha = alpha
        self.X_original = X
        self.X = np.hstack((np.ones( (self.ndata, 1)), (X - np.mean(X, 0)) / np.std(X, 0)))
        self.y = y[:, np.newaxis]
        self.param = np.zeros((self.nvar+ 1, 1))
    
        
    def fit(self):
        Costs = np.zeros(self.iterations)

        for i in range(self.iterations):
            self.param = self.param -  (self.alpha/self.ndata)*(self.X.T@(self.X@self.param - self.y))
            
            
            cost = sum(((self.y-self.X@self.param)**2)/len(self.X))/ self.ndata
            Costs[i] = cost
        
        self.y_pred = self.X@self.param
        self.cost = Costs
        self.coef_ = self.param[1:]
        self.intercept_ = self.param[0]

        self.rescale()

        return self

    def rescale(self):
        Xmu = np.mean(self.X_original,axis= 0)
        Xstd = np.std(self.X_original,axis= 0)

        self.intercept_ =  np.array([np.round((self.intercept_ - np.divide(
                            self.coef_.T*Xmu,Xstd).sum())[0],6)])

        coeff = np.around(np.divide(self.coef_.T,Xstd),decimals= 6)[0]

        self.coef_ = coeff[:, np.newaxis]
        self.param = np.vstack((self.intercept_,self.coef_))

    def predict(self,X):
        sample_n = len(X)
        X_new = np.hstack( (np.ones((sample_n, 1)), X))
        y_pred = X_new @ self.param
        return y_pred

    def get_r_squared(self, X=None, y=None):

        if X!=None and y!=None:
            y_pred = predict(X)
        else:
            y_pred = self.y_pred
            y = self.y

        print(y_pred[:7])
        SST = ((y - y.sum())**2).sum()
        SSE = ((y - y_pred)**2).sum()
        return (1-(SSE/SST))

    def plot(self):
        plt.plot(range(len(self.cost)),self.cost)
        plt.title('Loss Function')
        plt.xlabel('iterations')
        plt.ylabel('Cost')
        plt.show()