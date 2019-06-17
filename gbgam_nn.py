import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from nan_model import nan_model
from keras.models import Sequential, Model
from keras.layers import Dense, PReLU, LeakyReLU, ELU, Add, Input, Dropout, BatchNormalization
from keras import optimizers
from keras.wrappers.scikit_learn import KerasRegressor

from gbgam import *


def make_nan_cols(array):
    nan_arr = np.isnan(array)
    means = np.nanmean(array, axis = 0)
    impute_arr = np.array(array, copy = True)
    for r in range(len(impute_arr)):
        for c in range(len(impute_arr[r])):
            if(np.isnan(impute_arr[r][c])):
                impute_arr[r][c] = means[c]
    return np.concatenate([impute_arr, nan_arr], axis = 1)


class keras_additive_regressor:
    def __init__(self, columns,
                 single_model, 
                 pair_model, 
                 k = 5,
                 ):
        
        self.single_models = dict()
        self.single_model_inputs = dict()
        self.column_pairs = []
        self.k = k
        self.pair_models = dict()
        self.columns = list(columns)
        self.pair_scores = []
        
        self.single_model_base = single_model
        self.pair_model_base = pair_model
        self.single_model = None
        self.minimum = 0
    
    
    def fit(self, x, y, epochs = 10, optimizer = 'adam', pair_scores = None, sample_weight = None, forbidden_interactions = list()):
        self.mimimum = np.min(y)
        
        single_model_arr = list()
        single_input_arr = list()
        for col in self.columns:
            inputs, sub_model = self.single_model_base(col)
            self.single_models[col] = sub_model
            self.single_model_inputs[col] = inputs
        
        
        
        output = Add()(list(self.single_models.values()))
        
        for key in self.single_models:
            self.single_models[key] = Model(inputs = self.single_model_inputs[key], output = self.single_models[key])
        
        self.single_model = Model(inputs = list(self.single_model_inputs.values()), outputs = output)
        self.single_model.compile(loss='mean_squared_error', optimizer='adam')
        
        inputs = []
        for col in self.single_model_inputs.keys():
            inputs.append(make_nan_cols(x[col].values.reshape(-1, 1)))
        self.single_model.fit(inputs, y, epochs = epochs, batch_size = 128)
        
        '''
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
                x_pair = make_nan_cols(x_pair)
                local_y = y - cache
                fitted = clone(self.pair_model_base).fit(x_pair, local_y, sample_weight = sample_weight)
                self.pair_models[(col1, col2)].append(fitted)
                cache = cache + fitted.predict(x_pair)'''
        
    def predict(self, x):   
        inputs = []
        for col in self.single_model_inputs.keys():
            inputs.append(make_nan_cols(x[col].values.reshape(-1, 1)))
        return np.maximum(self.minimum, self.single_model.predict(inputs))
    
    def predict_column_array(self, x, col):
        x_col = make_nan_cols(x.reshape(-1, 1))
        return self.single_models[col].predict(x_col)
    
    def predict_pair(self, x_pair, pair_index):
        predicts = self.pair_models[self.column_pairs[pair_index]].predict(make_nan_cols(x_pair))
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

def create_model1(name):
    # create model
    x = inputs = Input(shape=(2,))
    x = BatchNormalization()(x)
    old_x = None
    for i in range(7):
        x = Dense(256)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        prev_x = x = Dropout(0.25)(x)
        if(i > 0):
            x = Add()([x, old_x])
        old_x = prev_x
    x = Dense(1, activation='linear', name = name)(x)
    # Compile model
    #model.compile(loss='mean_squared_error', optimizer='adam')
    return inputs, x



# single_models = []
# single_inputs = []
# inputs, x = create_model1()
# single_models.append(x)
# single_inputs.append(inputs)
# inputs, x = create_model1()
# single_models.append(x)
# single_inputs.append(inputs)
# predictions = Add()(single_models)
# model = Model(inputs = single_inputs, outputs = predictions)
model = keras_additive_regressor(x_train.columns, create_model1, create_model1, 0)
sgd = optimizers.SGD(lr=0.03, decay=1e-6, momentum=0.9, nesterov=True)
model.fit(x_train.rank(pct=True), y_train, epochs = 100, optimizer = sgd)