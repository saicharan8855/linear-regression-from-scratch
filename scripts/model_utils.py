"""
Model utilities for Linear Regression from scratch.

This module contains core helper functions used by the linear regression
engine, including train-test splitting, feature scaling, and bias handling.
"""

import numpy as np

# function for train test split
def train_test_split_np(X , y , test_size = 0.2 , random_state = None): 
    # giving random_state = none , python generates randomness without needing number  
    """
    split the train testsets using numpy only
    X : feature matrix
    y : target vector
    
    """
    # for seeding
    if random_state is not None:
        np.random.seed(random_state)

    # to calculate split size
    n_samples = X.shape[0]
    test_count = int(n_samples * test_size)

    # create and shuffle array

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    # split indices

    train_indices = indices[:-test_count]
    # the logic is for example n_samples = 100 -> [start : end - 20(test split = 0.2)] =80  
    test_indices = indices[-test_count:] # [80 : end] this starts where (end - testsplit) 


    # the above tain_indice and test_indice gets applied to this just like dictionary dispatch
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test



# function for standardization
def standardize_train_test(X_train,X_test):
    """
    the formula for standardisation is (x' = x - mu / sigma ).
    This is a scaling step where for large scaled features we make their mean get fall for 0 and std for 1
    """
    mu = np.mean(X_train , axis = 0) # axis used to count mean along rows

    sigma = np.std(X_train , axis = 0)

    # what if some features have sigma or variance = 0
    sigma[sigma == 0] = 1

    X_train_scaled = (X_train - mu)/sigma

    X_test_scaled = (X_test - mu)/sigma

    return X_train_scaled,X_test_scaled , mu , sigma



# function for bias column x_0 = 1
def bias_term(X_train_scaled,X_test_scaled):
    rows_train = X_train_scaled.shape[0]  # no of rows in  X_train_scaled
    rows_test = X_test_scaled.shape[0]

    ones_column_train = np.ones((rows_train,1))  # generates a column of one with (rows = rows_train , col = 1)
    ones_column_test = np.ones((rows_test,1))

    # stacking ones column and x_scaled horizontally
    h_stack_train = np.hstack((ones_column_train , X_train_scaled)) 
    h_stack_test = np.hstack((ones_column_test , X_test_scaled))
     

    return h_stack_train,h_stack_test


# function for r^2 score
def r2_score(y_true , y_pred):

    ss_res = np.sum((y_true - y_pred)**2)  # SSR
    ss_tot = np.sum((y_true - np.mean(y_true))**2)  # SST
    return 1 - (ss_res/ss_tot)


# function for rmse
def rmse(y_true , y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))  # calculates average prediction error in target units

    
    
    














