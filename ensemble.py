
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import math

def load_data(dataset):
    """
    Load a pair of data X,y 

    Params
    ------
    dataset:    train/valid/test

    Return
    ------
    X:          shape (N, 240)
    y:          shape (N, 1)
    """
    X = pd.read_csv(f"../petfinder-pawpularity-score/{dataset}.csv", header=None).to_numpy()
    #y = pd.read_csv(f"../petfinder-pawpularity-score/{dataset}.csv", header=None).to_numpy()
    y = X[:, len(X[0]) - 1]

    # delete the first row of header
    X = np.delete(X, 0, 0)
    # delete the first column of header, ID, and the last column, pawpularity
    X = np.delete(X, 0, 1)
    X = np.delete(X, 12, 1)

    # delete the header
    y = np.delete(y, 0, 0)
    
    # convert string to int
    X = X.astype(dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    

    return X_train, y_train, X_test, y_test



def main():
    np.random.seed(0)
    train_X, train_y, test_X, test_y = load_data("train")
    
    n_samples, n_features = train_X.shape

    N = 150 # Each part will be tried with 1 to 150 estimators
    rf_error_1 = []
    rf_error_2 = []
    rf_error_3 = []
    # for i in range(N):
    # # Train RF with m = sqrt(n_features) recording the errors (errors will be of size 150)
    #     clf = RandomForestClassifier(n_estimators = i + 1, max_features = int(np.sqrt(n_features)))
    #     clf.fit(train_X, train_y)
    #     pred = clf.predict(test_X)
    #     rf_error_1.append(round(1 - metrics.accuracy_score(test_y, pred), 2))

    # Train RF with m = n_features recording the errors (errors will be of size 150)
    # pred = None
    clf = RandomForestClassifier(n_estimators = 1000, max_features = int(np.sqrt(n_features)))
    clf.fit(train_X, train_y)
    predict = clf.predict(test_X)
    
    rmse = math.sqrt(mean_squared_error(test_y, predict))
    print(rmse)

    # Train RF with m = n_features/10 recording the errors (errors will be of size 150)
    # for i in range(N):
    #     clf = RandomForestClassifier(n_estimators = i + 1, max_features = int(n_features / 3))
    #     clf.fit(train_X, train_y)
    #     pred = clf.predict(test_X)
    #     rf_error_3.append(round(1 - metrics.accuracy_score(test_y, pred), 2))
    
    ####################
    ab_error_1 = []
    ab_error_2 = []
    ab_error_3 = []

    # Train AdaBoost with max_depth = 1 recording the errors (errors will be of size 150)

    # clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 1), n_estimators = 150, learning_rate = 0.1)
    # clf.fit(train_X, train_y)
    # predict = clf.predict(test_X)
    # square_error = 0
    # for i in range(len(predict)):
    #     error = int(test_y[i]) - int(predict[i])
    #     square_error += pow(error, 2)
    # rmse = math.sqrt(square_error / len(predict))
    # print(rmse)

    # # Train AdaBoost with max_depth = 3 recording the errors (errors will be of size 150)
    # for i in range(N):
    #     clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 3), n_estimators = i + 1, learning_rate = 0.1)
    #     clf.fit(train_X, train_y)
    #     pred = clf.predict(test_X)
    #     ab_error_2.append(round(1 - metrics.accuracy_score(test_y, pred), 2))

    # # Train AdaBoost with max_depth = 5 recording the errors (errors will be of size 150)
    # for i in range(N):
    #     clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 5), n_estimators = i + 1, learning_rate = 0.1)
    #     clf.fit(train_X, train_y)
    #     pred = clf.predict(test_X)
    #     ab_error_3.append(round(1 - metrics.accuracy_score(test_y, pred), 2))

if __name__ == '__main__':
    main()