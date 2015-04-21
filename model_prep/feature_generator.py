import numpy as np
import pandas as pd
from matplotlib import finance
import datetime


def get_basic_records(tickers, d1, d2, option):

    """

    This function is to get daily basic records.

    Args:
    tickers (list) : a list containing company tickers
    d1 (datetime) : start date
    d2 (datetime) : end date
    option (string) : a string of feature name. 
        Available: "open","close","volume","return","return rate".

    Return:
    results (array) : an numpy array containing requested features.
        Array with shape m,d.
        m: number of instances; d: number of features.

    """
    
    quotes = [finance.quotes_historical_yahoo(ticker, d1, d2, asobject=True) for ticker in tickers]
    
    if option == "open":
        results = np.array([q.open for q in quotes]).astype(np.float) 
    
    elif option == "close":
        results = np.array([q.close for q in quotes]).astype(np.float)
        
    elif option == "volume":
        results = np.array([q.volume for q in quotes]).astype(np.float)
        
    elif option == "return":
        opens = np.array([q.open for q in quotes]).astype(np.float)
        closes = np.array([q.close for q in quotes]).astype(np.float)
        results = closes - opens
    
    elif option == "return rate":
        opens = np.array([q.open for q in quotes]).astype(np.float)
        closes = np.array([q.close for q in quotes]).astype(np.float)
        results = (closes - opens)/opens
    
    else:
        print "Option Not Found!"
        results = None
        
    return results



def get_lower_freq(daily_data, days, option):

    """

    This function is to get lower frequency data to reduce the dimemsion of features. 

    Args:
    daily_data (array) : an array containing daily features.
    days (int) : days of combination. Generally 5 (1 week) or 10 (2 weeks).
    option (string) : a string of feature name.
        Available: "open", "close", "volume", "return", "return rate"

    Return:
    results (array) : an array containing requested features in lower frequency. 
        Array with shape m, d
        m: number of instances; d: number of features.

    """
    
    n = days
    
    if (option == "open") or (option == "close"):
        results = []
        for row in daily_data:
            LFD = []
            for i in range(0,len(row),n):
                LFD.append(row[i])
            results.append(LFD)
        results = np.array(results)
    
    elif option == "volume":
        results = []
        for row in daily_data:
            LFD = []
            for i in range(0,len(row)-n+1,n):
                s = 0
                for j in range(i,i+n):
                    s += row[j]
                aver = s/float(n)
                LFD.append(aver)
            results.append(LFD)
        results = np.array(results)
    
    elif option == "return":
        results = []
        for row in daily_data:
            LFD = []
            for i in range(0, len(row)-n+1, n):
                r = row[i+n] - row[i]
                LFD.append(r)
            results.append(LFD)
        results = np.array(results)
    
    elif option == "return rate":
        results = []
        for row in daily_data:
            LFD = []
            for i in range(0, len(row)-n+1, n):
                r = row[i+n] - row[i]
                rr = r/float(row[i])
                LFD.append(rr)
            results.append(LFD)
        results = np.array(results)
    
    else:
        print "Option Not Found!"
        results = None
        
    return results



def get_sequential_data(daily_data, days, option):
    
    """

    This function is to get sequential features such as Moving Average and Volatility.

    Args:
    daily_data (array) : an array containing daily features.
    days (int) : days of range. Generally 5 (1 week) or 10 (2 weeks).
    option (string) : a string of feature name.
        Available: "moving average", "volatility".

    Return:
    results (array) : an array containing requested features.
        Array with shape m, d
        m: number of instances; d: number of features.

    """

    n = days
    
    if option == "moving average":
        results = []
        for row in daily_data:
            SMA = []
            for i in range(len(row)-n+1):
                s = 0
                for j in range(i,i+n):
                    s += row[j]
                aver = s/float(n)
                SMA.append(aver)
            results.append(SMA)
        results = np.array(results)
    
    elif option == "volatility":
        results = []
        for row in daily_data:
            SV = []
            for i in range(len(row)-n+1):
                records = []
                for j in range(i, i+n):
                    records.append(row[j])
                std = np.std(records)
                SV.append(std)
            results.append(SV)
        results = np.array(results)
    
    else:
        print "Option Not Found!"
        results = None
                
    return results

def normalize_data(data):
    data_mean = data.mean(axis=1)
    data_std = data.std(axis=1)
    data_mean = data_mean.reshape(len(data_mean),1)
    data_std = data_std.reshape(len(data_std),1)
    normalized_data = ((data-data_mean)/data_std)
    return normalized_data




