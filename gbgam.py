import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.tree import DecisionTreeRegressor
from sklearn import model_selection
from sklearn import metrics
from sklearn.utils import resample
from sklearn.base import clone
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt


class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_min = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
        normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [normalized_min, normalized_mid, normalized_max]
        return sp.ma.masked_array(sp.interp(value, x, y))

def cv_evaluate(model, x, y, folds = 5, sample_weight = None):
    kf = model_selection.KFold(n_splits=folds, shuffle=True, random_state=123)
    score = 0
    for index, (dev_index, val_index) in enumerate(kf.split(x)):
        dev_x, val_x = x.iloc[dev_index,:], x.iloc[val_index,:]
        dev_y, val_y = y[dev_index], y[val_index]
        
        pred = model.fit(dev_x, dev_y).predict(val_x)
        score = score + metrics.mean_squared_error(val_y, pred) / folds
    return score
class additive_regressor:
    def __init__(self, columns, k = 5, lr = 0.1,
                 single_model = DecisionTreeRegressor(max_depth=2, min_samples_leaf=60, random_state = 123), 
                 pair_model =  DecisionTreeRegressor(max_depth=2, min_samples_leaf=60, random_state = 123),
                 ):
        
        self.models = dict()
        self.column_pairs = []
        self.k = k
        self.pair_models = dict()
        self.columns = list(columns)
        self.pair_scores = []
        
        self.single_model_base = single_model
        self.pair_model_base = pair_model
        self.bias = 0
        self.iters = 0
        self.lr = lr
        self.minimum = 0
    
    def sum_predict_all(self, x):
        predicts = np.full(len(x), self.bias)
        
        for col in self.models:
            x_col = x[col].values.reshape(-1, 1)
            for i, model in enumerate(self.models[col]):
                predicts = predicts + model.predict(x_col)*self.lr
        
        for (col1, col2) in self.pair_models:
            x_pair = x[[col1, col2]].values
            for i, model in enumerate(self.pair_models[(col1, col2)]):
                predicts = predicts + model.predict(x_pair)*self.lr
        
        return predicts
    
    def fit(self, x, y, iters = 10, iters_pair = None, pair_scores = None, sample_weight = None, forbidden_interactions = list()):
        self.iters = iters
        self.bias = np.mean(y)
        self.mimimum = np.min(y)
        
        if iters_pair == None:
            iters_pair = iters
        self.iters_pair = iters_pair
        cache = np.full(len(y), self.bias)
        
        for col in self.columns:
            self.models[col] = []
        
        for iteration in range(iters):
            for col in self.columns:
                local_y = y - cache
                x_col = x[col].values.reshape(-1, 1)
                fitted = clone(self.single_model_base).fit(x_col, local_y, sample_weight = sample_weight)
                self.models[col].append(fitted)
                cache = cache +  self.lr * fitted.predict(x_col)
        
        if(pair_scores == None):
            pair_scores = []
            local_y = y - cache
            for i in range(len(self.columns)):
                for j in range(i + 1, len(self.columns)):
                    col1 = self.columns[i]
                    col2 = self.columns[j]
                    if (col1 in forbidden_interactions) or (col2 in forbidden_interactions):
                        continue
                    x_pair = x[[col1, col2]]
                    pair_scores.append( (cv_evaluate(
                                                    clone(self.pair_model_base),
                                                    x_pair, local_y, sample_weight = sample_weight),
                                        col1, col2))
        
        self.pair_scores = pair_scores = sorted(pair_scores)
        print(pair_scores)
        
        for i in range( min(self.k,len(pair_scores)) ):
            col1 = pair_scores[i][1]
            col2 = pair_scores[i][2]
            self.column_pairs.append( (col1, col2) )
            self.pair_models[(col1, col2)] = []
        
        for iteration in range(iters_pair):
            for i in range( min(self.k,len(pair_scores)) ):
                col1 = pair_scores[i][1]
                col2 = pair_scores[i][2]
                x_pair = x[[col1, col2]].values
                local_y = y - cache
                fitted = clone(self.pair_model_base).fit(x_pair, local_y, sample_weight = sample_weight)
                self.pair_models[(col1, col2)].append(fitted)
                cache = cache + self.lr * fitted.predict(x_pair)
        
    def predict(self, x):   
        return np.maximum(self.minimum, self.sum_predict_all(x))
    
    def predict_column_array(self, x, col):
        x_col = x.reshape(-1, 1)
        predicts = np.zeros(len(x))
        for i, model in enumerate(self.models[col]):
            predicts = predicts + model.predict(x_col) * self.lr
        return predicts
    
    def predict_pair(self, x_pair, pair_index):
        predicts = np.zeros(len(x_pair))
        col1, col2 = self.column_pairs[pair_index]
        for i, model in enumerate(self.pair_models[(col1, col2)]):
            predicts = predicts + model.predict(x_pair) * self.lr
        return predicts
    
    def plot(self, x, col, ax = None, spaces = 200, target = None):
        minimum = x[col].quantile(0.05)
        maximum = x[col].quantile(0.95)
        spacing = (maximum - minimum)/spaces
        x_test = np.arange(minimum, maximum, spacing)[:, np.newaxis]
        if(ax is None):
            ax = plt.axes()
        ax.plot(x_test, self.predict_column_array(x_test, col))
        if(target is not None):
            ax.scatter(x[col], target)
        ax.axhline(0, color='black', lw=0.75, alpha = 0.75)
        return ax
    
    def plot_pairs(self, x, pair_index, ax = None, spaces_1 = 100, spaces_2 = 100, target = None, dot_size = 10):
        col1, col2 = self.column_pairs[pair_index]
        
        minimum1 = x[col1].quantile(0.025)
        maximum1 = x[col1].quantile(0.975)
        spacing1 = (maximum1 - minimum1)/spaces_1
        
        minimum2 = x[col2].quantile(0.025)
        maximum2 = x[col2].quantile(0.975)
        spacing2 = (maximum2 - minimum2)/spaces_2
        
        x_test = np.mgrid[minimum1:maximum1:spacing1, minimum2:maximum2:spacing2]
        coords1 = x_test[0].flatten()
        coords2 = x_test[1].flatten()
        
        x_test = np.array([coords1,coords2]).T
        if(ax is None):
            ax = plt.axes()
        result = self.predict_pair(x_test, pair_index)
        
        vmin = abs(result.min())
        vmax = abs(result.max())
        
        scatter = ax.scatter(coords1, coords2, marker = 's', c = result, s = dot_size , cmap = 'bwr', vmin = -max(vmin, vmax), vmax = max(vmin, vmax))
        ax.set_xlabel(col1)
        ax.set_ylabel(col2)
        return ax, scatter
    def bootstrap(self, x, y, subsample = 0.5, iterations = 100, sample_weight = None):
        bootstrap_models = [] 
        for i in tqdm(range(iterations)):
            model = additive_regressor(self.columns, k = self.k, single_model = self.single_model_base, pair_model = self.pair_model_base)
            x_train, y_train, weight_train = None, None, None
            if(sample_weight == None):
                x_train, y_train  = resample(x, y, n_samples = int(subsample*len(y)) )
            else:
                x_train, y_train, weight_train  = resample(x, y, sample_weight, n_samples = int(subsample*len(y)) )
            model.fit(x_train, y_train, self.iters, self.iters_pair, self.pair_scores, sample_weight = weight_train)
            bootstrap_models.append(model)
        return bootstrap_models



def save_model(model, file):
    with open(file, 'wb') as pickle_file:
        pickle.dump(model, pickle_file)
    
def load_model(file):
    with open(file, 'rb') as fin:
        return pickle.load(fin)
    
def bootstrap_predict_column(models, x, col):
    predictions = [] 
    for model in models:
        predictions.append(model.predict_column_array(x, col))
    return predictions

def bootstrap_predict_pair(models, x_pair, pair_index):
    predictions = [] 
    for model in models:
        predictions.append(model.predict_pair(x_pair, pair_index))
    return predictions

def bootstrap_predict(models, x):
    predictions = [] 
    for model in models:
        predictions.append(model.predict(x))
    return np.mean(predictions, axis = 0)
def bootstrap_plot(models, x, col, ax = None, spaces = 200, lower = 5, upper = 95):
    minimum = x[col].quantile(0.05)
    maximum = x[col].quantile(0.95)
    spacing = (maximum - minimum)/spaces
    x_test = np.arange(minimum, maximum, spacing)[:, np.newaxis]
    predictions = np.array(bootstrap_predict_column(models, x_test, col))
    if(ax is None):
        ax = plt.axes()
    ax.plot(x_test, np.mean(predictions, axis = 0))
    ax.axhline(0, color='black', lw=0.75, alpha = 0.75)
    ax.fill_between(x_test[:,0], np.percentile(predictions,lower, axis = 0), np.percentile(predictions,upper, axis = 0), alpha = 0.25)
    return ax

def bootstrap_plot_pair(models, x, pair_index, ax = None, spaces_1 = 100, spaces_2 = 100, dot_size = 10):
    col1, col2 = models[0].column_pairs[pair_index]
        
    minimum1 = x[col1].quantile(0.01)
    maximum1 = x[col1].quantile(0.99)
    spacing1 = (maximum1 - minimum1)/spaces_1
        
    minimum2 = x[col2].quantile(0.01)
    maximum2 = x[col2].quantile(0.99)
    spacing2 = (maximum2 - minimum2)/spaces_2
      
    x_test = np.mgrid[minimum1:maximum1:spacing1, minimum2:maximum2:spacing2]
    coords1 = x_test[0].flatten()
    coords2 = x_test[1].flatten()
        
    x_test = np.array([coords1,coords2]).T
    if(ax is None):
        ax = plt.axes()
    result = np.mean(bootstrap_predict_pair(models, x_test, pair_index), axis = 0)
        
    vmin = abs(result.min())
    vmax = abs(result.max())
      
        
    scatter = ax.scatter(coords1, coords2, marker = 's', c = result, s = dot_size, cmap = 'bwr', vmin = -max(vmin, vmax), vmax = max(vmin, vmax))
    ax.set_xlabel(col1)
    ax.set_ylabel(col2)
    return ax, scatter