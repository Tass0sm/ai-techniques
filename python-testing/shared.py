import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

def prepare_data(frame):
    X = frame.drop(['Id', 'SalePrice'], axis=1)
    y = frame['SalePrice']

    for col in X.columns:
        X[col] = X[col].astype('category').cat.codes

    return X, y

def print_results(y_hat, y):
    print("Mean Absolute Error: ", np.mean(np.absolute(y_hat - y)))
    print("Mean Absolute Percent Error: ",
          np.mean(np.absolute((y_hat - y) / y)))
    print("Median Absolute Error: ", np.median(y_hat - y))
    print("R^2 Value: ", r2_score(y, y_hat))
