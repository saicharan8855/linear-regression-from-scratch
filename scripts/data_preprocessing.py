# Phase III: Ames Housing Data Preparation
import numpy as np
import pandas as pd

# in this library file we just write our functions later on we import this into notebooks
# read csv
def load_raw_data(train_path,test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df,test_df


# NOTE:
"""
the log transformation of saleprice column is applied based in well established domain knowledge
in housing economics where property prices are known to be approximately log-normally distributed.

This step is not assumed blindly , its validity is verified later through residual logistics
and model validation analyses
"""


def log_transform(train_df):
    """
    This function takes the input train_df ,
    extracts saleprice column from the training dataframe
    and applies a natural logarithm transformation to the target.

    Thus the skewness is reducedin house prices and makes the gaussian noise assumption
    inlinear regression more appropriate.
    """
    x = train_df["SalePrice"]
    y = np.log(x)   # (np.log > math.log) coz it is optimized for vectorized operations on large datasets
    return y


def num_or_cat(train_df):
    """
    This function here is created to check whether the column is categorical or numerical
    so we use the function 'pd.api.types.is_numerical_dtype()'  a utility in the pandas library for checking  
    if  a given array, pandas Series, or a Numpy dtype object is considered a numeric data type.

    we give train_df[col] ->  Looks up actual data for that column name to check its dtype
    because column name != column data..so it iterates and check the variables in the columns.

    """
    numerical = []
    categorical = []
    
    for i in train_df.columns:
        if pd.api.types.is_numeric_dtype(train_df[i]):
            numerical.append(i)
        else:
            categorical.append(i)
    return numerical,categorical
    

# function to fill Nans with median of that column
def nan_check(train_df):
    """
    someone may get a doubt about median imputation like what if nans are between noticable and more.
    my answer is:
    We use median imputation as a basic first step to fill in missing data and create a full dataset for
    analysis. Later, we check how the missing data affects results during model checks and error reviews.
    
    """
    train_df = train_df.copy()
    numerical = []
    for cols in train_df.columns:
        if pd.api.types.is_numeric_dtype(train_df[cols]):
            numerical.append(cols)

    for cols in numerical:
        median = train_df[cols].median()
        train_df[cols] = train_df[cols].fillna(median)

    return train_df


# function to fill None in categorical columns 
def none_check_cat(train_df):

    """
    The question rises here : “why not fill categorical nans with mode? why fill with the string 'None' ?

    My answer is :
    For categorical data with missing values, we fill them using a special category called "None." This keeps 
    track of the absence as real information, instead of using the most common category (mode) -> which might 
    create misleading patterns.
    
    """
    train_df = train_df.copy()
    categorical = []
    for cols in train_df.columns:
        if not pd.api.types.is_numeric_dtype(train_df[cols]):
            categorical.append(cols)

    for cols in categorical:   
        train_df[cols] = train_df[cols].fillna("None")

    return train_df

# function for correlation
def compute_numeric_correlations(X_df,y):
    correlations = {}
    """
    x_df = cleaned data frame excluding target feature 'y'
    y = log_transformed saleprice target variable
    """
    for col in X_df.columns:
        if pd.api.types.is_numeric_dtype(X_df[col]):
            corr_value = X_df[col].corr(y)
            correlations[col] = corr_value 

    # orient = "index" means key -> row index and value -> column in correlations dictionary
    corr_df = (
        pd.DataFrame.from_dict(correlations, orient="index", columns=["correlation"]) 
        .assign(abs_correlation=lambda df: df["correlation"].abs())
        .sort_values(by="abs_correlation", ascending=False)
    )
    
    return corr_df


# select top k features
def top_k(corr_df , k = 10):
    top_features = corr_df.iloc[:k].index.tolist() # since the columns are already sorted upwards we dont need to write loop inside
    return top_features



# onehot encode for categorical
def categorical_cols(X_df):
    categorical = []
    for col in X_df.columns:
        if not pd.api.types.is_numeric_dtype(X_df[col]):
            categorical.append(col)

    return categorical


def onehot_encode_categorical(X_df,categorical):
    X_df = X_df.copy()

    encoded_df = pd.get_dummies(X_df[categorical],drop_first = False)

    X_numeric = X_df.drop(columns = categorical)

    X_final = pd.concat([X_numeric , encoded_df] , axis = 1)

    return X_final
    
    
        





    