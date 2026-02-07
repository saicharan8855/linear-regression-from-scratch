"""
Linear Regression implementation from scratch using NumPy.

This module defines the LinearRegressionMaster class, supporting
ordinary least squares and gradient-based optimization.
"""


import numpy as np

class LinearRegressionMaster():
    def __init__(self):
        self.theta = None
        self.loss_history = []
        
    def predict(self,X):
        return X @ self.theta  # @ ==  np.dot
        
    def compute_cost(self,X,y):
        y_hat = self.predict(X)
        residual = y_hat - y
        r_squared = residual**2
        m = y.shape[0]
        # We use the standard squared-error loss scaled by 1/2m which differs MSE by a constant factor
        cost = 1/(2*m)*np.sum(r_squared)
        return cost

    def fit_ols(self,X,y):
        # compute xtx
        xtx = np.linalg.pinv(X.T @ X)
        xty = X.T @ y
        self.theta = xtx @ xty

    def fit_gradient_descent(self,X,y,alpha = 0.01,epochs = 1000): # alpha = learning rate , epochs = iterations

        m = y.shape[0]
        self.loss_history = [] # reset loss history for fresh training
        
        if self.theta is None:
            self.theta = np.zeros((X.shape[1]))  # initializing parameters

        for epoch in range(epochs):
            
            # 1.predict 
            y_hat = X @ self.theta

            # 2.residual 
            r = y_hat - y

            # 3.gradient 
            grad = (1/m) * (X.T @ r)

            # 4.update theta
            self.theta -= alpha * grad

            # 5.loss track
            loss = self.compute_cost(X,y)
            self.loss_history.append(loss)

            
        
     
        
    


