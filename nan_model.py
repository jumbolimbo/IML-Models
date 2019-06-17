import numpy as np
from sklearn.base import clone
from sklearn.base import BaseEstimator

def make_nan_cols(array):
    nan_arr = np.isnan(array)
    means = np.nanmean(array, axis = 0)
    impute_arr = np.array(array, copy = True)
    for r in range(len(impute_arr)):
        for c in range(len(impute_arr[r])):
            if(np.isnan(impute_arr[r][c])):
                impute_arr[r][c] = means[c]
    return np.concatenate([impute_arr, nan_arr], axis = 1)

class nan_model(BaseEstimator):
    def __init__(self, base_model):
        self.model = clone(base_model)
    def fit(self, x, y, sample_weight = None):
        if(sample_weight != None):
            self.model = self.model.fit(x, y, sample_weight = sample_weight)
        else:
            self.model = self.model.fit(x, y)
    def predict(self, x):
        return self.model.predict(x)