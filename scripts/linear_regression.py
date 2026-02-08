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

    def fit_ridge(self, X, y, lambda_):
        n_features = X.shape[1]

        # identity matrix
        I = np.eye(n_features)

        # dont regularize bias term
        I[0, 0] = 0

        # Ridge closed-form solution
        xtx = X.T @ X
        ridge_term = lambda_ * I

        self.theta = np.linalg.pinv(xtx + ridge_term) @ (X.T @ y)

    def fit_lasso(self, X, y, lambda_, epochs=1000):
       
        m, n = X.shape

    # initialize theta if needed
        if self.theta is None:
            self.theta = np.zeros(n)

        self.loss_history = []

        for _ in range(epochs):
            for j in range(n):

                # skip bias term (do NOT regularize)
                if j == 0:
                    r = y - (X @ self.theta) + self.theta[j] * X[:, j]
                    self.theta[j] = np.sum(X[:, j] * r) / np.sum(X[:, j] ** 2)
                    continue

                # partial residual
                r = y - (X @ self.theta) + self.theta[j] * X[:, j]

                rho = np.sum(X[:, j] * r)

                # soft-thresholding
                if rho < -lambda_ / 2:
                    self.theta[j] = (rho + lambda_ / 2) / np.sum(X[:, j] ** 2)
                elif rho > lambda_ / 2:
                    self.theta[j] = (rho - lambda_ / 2) / np.sum(X[:, j] ** 2)
                else:
                    self.theta[j] = 0.0

            # track loss
            loss = self.compute_cost(X, y)
            self.loss_history.append(loss)

    


            
        
     
        
    


