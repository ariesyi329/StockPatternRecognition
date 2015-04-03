import numpy as np
import pandas as pd
from matplotlib import finance
import datetime


#function to load tickers into lists
def load_ticker_data(dataset):
    
    """
    Args:
    dataset(string): 
        if "train", then loading training data
        if "test", then loading test data

    Return:
    F,IT,C,I (list): four lists of tickers

    """
    
    F = list(pd.read_csv('../tickers/Financials_{}.txt'.format(dataset),header=None)[0])
    IT = list(pd.read_csv('../tickers/Information Technology_{}.txt'.format(dataset),header=None)[0])
    C = list(pd.read_csv('../tickers/Consumer Discretionary_{}.txt'.format(dataset),header=None)[0])
    I = list(pd.read_csv('../tickers/Industrials_{}.txt'.format(dataset),header=None)[0])
    
    return F, IT, C, I


#construct y labels
def construct_y(F,IT,C,I):
    
    """
    Args:
    F,IT,C,I (list) are ticker lists for four domains.

    Return:
    F_y,IT_y,C_y,I_y (array) are 4 arrays with y labels.
        They are numpy arrays with shape m,d.
        m: number of instances; d: 1.
    """
    
    F_y = np.array([[0]*len(F)]).T
    IT_y = np.array([[1]*len(IT)]).T
    C_y = np.array([[2]*len(C)]).T
    I_y = np.array([[3]*len(I)]).T
    
    return F_y, IT_y, C_y, I_y

def shuffle_data(X,y):

    """
    Args:
    X (array): X dataset
    y (array): y dataset

    Return:
    data (array): shuffled dataset (combined X and y)
    """

    data = np.concatenate((X,y),axis=1)
    np.random.shuffle(data)
    return data

def finalize_data(F_X, IT_X, C_X, I_X, F_y, IT_y, C_y, I_y):

    X = np.concatenate((F_X,IT_X,C_X,I_X),axis=0) 
    y = np.concatenate((F_y,IT_y,C_y,I_y),axis=0)
    data = shuffle_data(X,y)
    X = data[:,:-1]
    y = data[:,-1]

    return X, y


