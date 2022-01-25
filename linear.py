import pandas as pd
import numpy as np

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit


import matplotlib.pyplot as plt
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
    y = y.astype(dtype=int)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=0)

    sss = StratifiedShuffleSplit(n_splits=1, train_size=0.8, random_state=0)
    sss.get_n_splits(X, y)
    for train_index, test_index in sss.split(X, y):
        new_X, new_Y = X[train_index], y[train_index]
        test_X, test_Y = X[test_index], y[test_index]

    sss = StratifiedShuffleSplit(n_splits=1, train_size=0.9, random_state=0)
    sss.get_n_splits(new_X, new_Y)
    for train_index, test_index in sss.split(new_X, new_Y):
        train_X, train_Y = new_X[train_index], new_Y[train_index]
        valid_X, valid_Y = new_X[test_index], new_Y[test_index]

    return train_X, test_X, valid_X, train_Y, test_Y, valid_Y

def score(model, X, y):
    """
    Score the model with X, y

    Params
    ------
    model:  the model to predict with
    X:      the data to score on
    y:      the true value y

    Return
    ------
    mse:    the mean square error
    """
    pred = model.predict(X)
    return mean_squared_error(y, pred)


def hyper_parameter_tuning(model_class, param_grid, train_X, train_y, valid_X, valid_y, test_X, test_y):
    best_mse = float('inf')
    best_param = None
    # Set up the parameter grid
    param_grid = list(ParameterGrid(param_grid))
    regression = None

    if (model_class == LinearRegression):
        # train the model with each parameter setting in the grid
        for i in range(len(param_grid)):
            regression = LinearRegression(**param_grid[i])
            regression.fit(train_X, train_y)

        # choose the model with lowest MSE on validation set
            mse = score(regression, valid_X, valid_y)
            if mse < best_mse:
                best_mse = mse
                best_param = param_grid[i]
        # then fit the model with the training and validation set (refit)
        new_data_X = np.concatenate((train_X, valid_X), axis=0)
        new_data_Y = np.concatenate((train_y, valid_y), axis=0)

        # return the fitted model and the best parameter setting
        regression = LinearRegression(**best_param)
        regression.fit(new_data_X, new_data_Y)
        print("Linear Regression MSE: ", best_mse)
        # return regression, best_param
    elif (model_class == Ridge):
        for i in range(len(param_grid)):
            regression = Ridge(**param_grid[i])
            regression.fit(train_X, train_y)
            mse = score(regression, valid_X, valid_y)
            #print(mae, " ", param_grid[i])
            if mse < best_mse:
                best_mse = mse
                best_param = param_grid[i]
        new_data_X = np.concatenate((train_X, valid_X), axis=0)
        new_data_Y = np.concatenate((train_y, valid_y), axis=0)
        regression = Ridge(**best_param)
        regression.fit(new_data_X, new_data_Y)
        print("Ridge Regression MSE: ", best_mse)
    elif (model_class == Lasso):
        # find best parameter
        for i in range(len(param_grid)):
            regression = Lasso(**param_grid[i])
            regression.fit(train_X, train_y)
            mse = score(regression, valid_X, valid_y)
            if mse < best_mse:
                best_mse = mse
                best_param = param_grid[i]
        # combine train and valid set
        new_data_X = np.concatenate((train_X, valid_X), axis=0)
        new_data_Y = np.concatenate((train_y, valid_y), axis=0)

        # train new set and predict test set
        regression = Lasso(**best_param)
        regression.fit(new_data_X, new_data_Y)
        # predict = regression.predict(test_X)
        # square_error = 0
        # for i in range(len(predict)):
        #     error = int(test_y[i]) - int(predict[i])
        #     square_error += pow(error, 2)
        # rmse = math.sqrt(square_error / len(predict))
        print(math.sqrt(score(regression, test_X, test_y)))
        #print("Lasso Regression MSE: ", best_mse)
    else:
        pass
    
    return regression, best_param

def plot_mae_alpha(model_class, params, train_X, train_y, valid_X, valid_y, test_X, test_y, title="Model"):
    """
    Plot the model MAE vs Alpha (regularization constant)

    Params
    ------
    model_class:    The model class to fit and plot
    params:         The best params found 
    train:          The training dataset
    valid:          The validation dataest
    test:           The testing dataset
    title:          The plot title

    Return
    ------
    None
    """
    # train_X = np.concatenate([train_X, valid_X], axis=0)
    # train_y = np.concatenate([train_y, valid_y], axis=0)

    # # set up the list of alphas to train on
    # alpha = [0.01, 0.1, 0.4, 0.6, 0.8, 1.0, 2.0, 5.0, 10.0]
    # # train the model with each alpha, log MAE
    # all_mae = []
    # index = 0
    # if (model_class == Ridge):
    #     for i in range(len(alpha)):
    #         params['alpha'] = alpha[i]
    #         regression = Ridge(**params)
    #         regression.fit(train_X, train_y)
    #         predict = regression.predict(test_X)
    #         all_mae.append(score(regression, test_X, test_y))
    # else:
    #     regression = Lasso(**params)
    #     regression.fit(train_X, train_y)
    #     predict = regression.predict(test_X)
    #     square_error = 0
    #     for i in range(len(predict)):
    #         error = int(test_y[i]) - int(predict[i])
    #         square_error += pow(error, 2)
    #     rmse = math.sqrt(square_error / len(predict))
    #     print(rmse)
    #     all_mae.append(score(regression, test_X, test_y))
            

def main():
    """
    Load in data
    """
    X_train, X_test, X_valid, y_train, y_test, y_valid = load_data("train")
    # valid = load_data('valid')
    # test = load_data('test')

    """
    Define the parameter grid each each classifier
    e.g. lasso_grid = dict(alpha=[0.1, 0.2, 0.4],
                           max_iter=[1000, 2000, 5000])
    """
    # Tune the hyper-paramter by calling the hyper-parameter tuning function
    # e.g. lasso_model, lasso_param = hyper_parameter_tuning(Lasso, lasso_grid, train, valid)
    #lasso_model, lasso_param = hyper_parameter_tuning(LinearRegression, lasso_grid, train, valid)

    # linear_grid = dict(positive=[True, False])
    # linear_model, linear_param = hyper_parameter_tuning(LinearRegression, linear_grid, X_train, y_train, X_valid, y_valid)
    # print(linear_model, linear_param)

    # Plot the MAE - Alpha plot by calling the plot_mae_alpha function
    # e.g. plot_mae_alpha(Lasso, lasso_param, train, valid, test, "Lasso")

    # ridge_grid = dict(alpha=[0.1, 0.2, 0.4], max_iter=[10000, 20000, 50000], tol=[1e-4, 1e-2, 1e-1, 1.0])
    # ridge_model, ridge_param = hyper_parameter_tuning(Ridge, ridge_grid, X_train, y_train, X_valid, y_valid)
    # print(ridge_model, ridge_param)
    # print()
    # plot_mae_alpha(Ridge, ridge_param, X_train, y_train, X_valid, y_valid, X_test, y_test, "Ridge")

    lasso_grid = dict(alpha=[0.001, 0.2, 0.4], max_iter=[5000, 2000, 5000], tol=[1e-4, 1e-2, 1e-1, 1.0], positive = [True, False])
    lasso_model, lasso_param = hyper_parameter_tuning(Lasso, lasso_grid, X_train, y_train, X_valid, y_valid, X_test, y_test)
    print(lasso_model, lasso_param)
    plot_mae_alpha(Lasso, lasso_param, X_train, y_train, X_valid, y_valid, X_test, y_test, "Lasso")


if __name__ == '__main__':
    main()
