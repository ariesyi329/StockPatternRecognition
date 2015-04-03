import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import finance
import datetime

def read_data(file_path):
    df = pd.read_csv(file_path, usecols = (0,1,2), names = ['ticker', 'company', 'sector'])
    return df

def select_ticker(df, sector):
	
	#select raw tickers in one industry
	sector_df = df[df['sector'] == sector]
	raw_tickers = list(sector_df['ticker'])

	#select tickers without missing data
	d1 = datetime.datetime(2005, 1, 1)
	d2 = datetime.datetime(2014, 12, 31)
	selected_tickers = []
	for ticker in raw_tickers:
		try:
			if len(finance.quotes_historical_yahoo(ticker, d1, d2)) == 2517:
				selected_tickers.append(ticker)
		except:
			pass
	
	return selected_tickers
	
def tickers_to_file(sector):
	df = read_data('data/Ticker_new.csv')
	selected_tickers = select_ticker(df, sector)
	filename = sector+'_tickers.txt'
    with open(filename, 'w') as out_file:
        for ticker in selected_tickers:
            out_file.write("%s\n" % ticker)
        
if __name__ == '__main__':
	sectors = ['Financials','Information Technology','Consumer Discretionary','Industrials']
	for sector in sectors:
		tickers_to_file(sector)
	
