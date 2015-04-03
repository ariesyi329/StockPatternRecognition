import numpy as np
import datetime
from sklearn.neighbors import KNeighborsClassifier
import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
from model_prep import feature_generator as fg
from model_prep import data_util as ut
from model_eval import supervised_eval as se

def sequential_data(dataset,d1,d2,days,option):
    
    #load tickers data
    F, IT, C, I = ut.load_ticker_data(dataset)
    
    #construct y labels
    F_y, IT_y, C_y, I_y = ut.construct_y(F, IT, C, I)
    
    #construct train_X
    F_X = fg.get_basic_records(F, d1, d2, "return rate")
    IT_X = fg.get_basic_records(IT, d1, d2, "return rate")
    C_X = fg.get_basic_records(C, d1, d2, "return rate")
    I_X = fg.get_basic_records(I, d1, d2, "return rate")
    F_X = fg.get_sequential_data(F_X, days, option)
    IT_X = fg.get_sequential_data(IT_X, days, option)
    C_X = fg.get_sequential_data(C_X, days, option)
    I_X = fg.get_sequential_data(I_X, days, option)
    #finalize train and test data
    X, y = ut.finalize_data(F_X,IT_X,C_X,I_X,
                            F_y,IT_y,C_y,I_y)

    return X, y


def sequential_model(X_train,y_train,X_test,y_test):
    
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(X_train, y_train)
    predicted = neigh.predict(X_test)
    
    return predicted

def main(days,option):
    d1 = datetime.datetime(2005, 1, 1)
    d2 = datetime.datetime(2014, 12, 31)
    train_X,train_y = sequential_data("train",d1,d2,days,option)
    test_X,test_y = sequential_data("test",d1,d2,days,option)
    predicted = sequential_model(train_X, train_y, test_X, test_y)
    se.evaluate_matrix(test_y,predicted)

if __name__ == "__main__":
    print "************10 day moving average*******"
    main(days=10,option="moving average")
    print "************20 day moving average*******"
    main(days=20,option="moving average")
    print "************10 day volatility**********"
    main(days=10,option="volatility")
    print "************20 day volatility**********"
    main(days=20,option="volatility")



