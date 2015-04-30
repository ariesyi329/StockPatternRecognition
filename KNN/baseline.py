import numpy as np
import datetime
from sklearn.neighbors import KNeighborsClassifier
import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
from model_prep import feature_generator as fg
from model_prep import data_util as ut
from model_eval import supervised_eval as se
from sklearn.svm import LinearSVC

def baseline_data(dataset,d1,d2,option):
    
    #load tickers data
    F, IT, C, I,tickers = ut.load_ticker_data(dataset)
    
    #construct y labels
    F_y, IT_y, C_y, I_y = ut.construct_y(F, IT, C, I)
    
    #construct train_X
    F_X = fg.get_basic_records(F, d1, d2, option)
    IT_X = fg.get_basic_records(IT, d1, d2, option)
    C_X = fg.get_basic_records(C, d1, d2, option)
    I_X = fg.get_basic_records(I, d1, d2, option)
    
    #finalize train and test data
    X, y,z = ut.finalize_data(F_X,IT_X,C_X,I_X,
                              F_y,IT_y,C_y,I_y,
                              tickers)

    X = X.astype(float)
    y = y.astype(float)
    y = y.astype(int)

    return X, y,z


def baseline_model(X_train,y_train,X_test,y_test):
    
    print X_train.shape

    feature_selection = LinearSVC(C=1, penalty="l1", dual=False)
    X_train_new = feature_selection.fit_transform(X_train, y_train)
    X_test_new = feature_selection.transform(X_test)

    
    print X_train_new.shape
    print X_test_new.shape

    neigh = KNeighborsClassifier(n_neighbors=4, p=1)
    neigh.fit(X_train_new, y_train)
    predicted = neigh.predict(X_test_new)
    
    return predicted

def main(option):
    d1 = datetime.datetime(2005, 1, 1)
    d2 = datetime.datetime(2014, 12, 31)
    train_X,train_y,train_tickers = baseline_data("training",d1,d2,option)
    test_X,test_y,test_tickers = baseline_data("validation",d1,d2,option)
    predicted = baseline_model(train_X, train_y, test_X, test_y)
    se.evaluate_matrix(test_y,predicted)

if __name__ == "__main__":
    main("return rate")



