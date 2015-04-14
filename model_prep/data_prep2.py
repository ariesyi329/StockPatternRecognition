import numpy as np
import data_util as ut

def main():
    F, IT, C, I = ut.load_ticker_data("tickers")
    tickers = F + IT + C + I
    np.random.shuffle(tickers)
    with open('../tickers/tickers.txt', 'w') as out_file:
        for ticker in tickers:
            out_file.write("%s\n" % ticker)

if __name__ == '__main__':
    main()



