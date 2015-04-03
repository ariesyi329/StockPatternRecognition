import numpy as np
import datetime
from sklearn.neighbors import KNeighborsClassifier
import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
from model_prep import feature_generator as fg
from model_prep import data_util as ut
from model_eval import supervised_eval as se

def low_freq_data(dataset,d1,d2,days):
    
    #load tickers data
    F, IT, C, I = ut.load_ticker_data(dataset)
    
    #construct y labels
    F_y, IT_y, C_y, I_y = ut.construct_y(F, IT, C, I)
    
    #construct train_X
    F_X = fg.get_basic_records(F, d1, d2, "open")
    IT_X = fg.get_basic_records(IT, d1, d2, "open")
    C_X = fg.get_basic_records(C, d1, d2, "open")
    I_X = fg.get_basic_records(I, d1, d2, "open")
    F_X = fg.get_lower_freq(F_X,days,option="return rate")
    IT_X = fg.get_lower_freq(IT_X,days,option="return rate")
    C_X = fg.get_lower_freq(C_X,days,option="return rate")
    I_X = fg.get_lower_freq(I_X,days,option="return rate")
    
    #finalize train and test data
    X, y = ut.finalize_data(F_X,IT_X,C_X,I_X,
                            F_y,IT_y,C_y,I_y)

    return X, y


def low_freq_model(X_train,y_train,X_test,y_test):
    
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(X_train, y_train)
    predicted = neigh.predict(X_test)
    
    return predicted

def main(days):
    d1 = datetime.datetime(2005, 1, 1)
    d2 = datetime.datetime(2014, 12, 31)
    train_X,train_y = low_freq_data("train",d1,d2,days)
    test_X,test_y = low_freq_data("test",d1,d2,days)
    predicted = low_freq_model(train_X, train_y, test_X, test_y)
    se.evaluate_matrix(test_y,predicted)

if __name__ == "__main__":
    print "**********5 day Return**********"
    main(days=5)
    print "**********10 day Return*********"
    main(days=10)



