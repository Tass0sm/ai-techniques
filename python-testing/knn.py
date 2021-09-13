import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from shared import *

np.random.seed(1)
full_frame = pd.read_csv('../data/train-and-test.csv')
frames = np.array_split(full_frame, 2)
train_X, train_y = prepare_data(frames[0]);
test_X, test_y = prepare_data(frames[1]);

regressor = KNeighborsRegressor(p=1, weights='distance', algorithm='brute')
regressor.fit(train_X, train_y)
predict_y = regressor.predict(test_X)

print_results(predict_y, test_y)
