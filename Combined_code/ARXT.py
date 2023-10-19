import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import det, inv
from scipy.special import gamma
from math import pi, ceil
from scipy.special import erfinv
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
# from statsmodels.tsa import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from random import randint
import warnings
from sklearn.metrics import mean_squared_error
from math import sqrt, log
from numpy.linalg import lstsq
import networkx as nx


warnings.filterwarnings("ignore")

def preprocessing(data, method):
    ts_train = []
    ts_valid = []
    ts_test = []
    ts_param = []
    flag_n = False
    flag_diff = False
    flag_dec = False
    if method is 'normalization':
        flag_n = True
    elif method is 'differencing':
        flag_diff = True
            
    for i in range(data.shape[1]):

        temp = list(data.iloc[6:][i].dropna())
        # print(temp)
        if len(temp) > 130:
            
            cut_off_1 = ceil(len(temp)*0.7)
            cut_off_2 = ceil(len(temp)*0.9)

            temp_train = temp[:cut_off_1]
            temp_val = temp[cut_off_1:cut_off_2]
            temp_test = temp[cut_off_2:]

            if flag_n is True:
                ts_param.append([np.mean(temp_train), np.std(temp_train), np.mean(temp_val), 
                np.std(temp_val), np.mean(temp_test), np.std(temp_test)])
                
                temp_train = (temp_train - np.mean(temp_train)) / np.std(temp_train)
                ts_train.append(temp_train)
                

                temp_val = (temp_val - np.mean(temp_val)) / np.std(temp_val)
                ts_valid.append(temp_val)

                temp_test = (temp_test - np.mean(temp_test)) / np.std(temp_test)
                ts_test.append(temp_test)
                

            elif flag_diff is True:

                temp_train_diff = [temp_train[i] - temp_train[i - 1] for i in range(1, len(temp_train))]
                temp_train_diff = [temp_train[0]] + temp_train_diff
                ts_train.append(temp_train_diff)

                temp_val_diff = [temp_val[i] - temp_val[i - 1] for i in range(1, len(temp_val))]
                temp_val_diff = [temp_val[0]] + temp_val_diff
                ts_valid.append(temp_val_diff)

                temp_test_diff = [temp_test[i] - temp_test[i - 1] for i in range(1, len(temp_test))]
                temp_test_diff = [temp_test[0]] + temp_test_diff
                ts_test.append(temp_test_diff)
                
    return ts_train, ts_valid, ts_test, ts_param

class AutoregressiveTree:
    
    def __init__(self, p, u0=0, alpha_u=1, X=None):
        self._X = X  # Exogenous data
        
        erf_temp = np.zeros([7,1])
        for i in range(1,8):
            erf_temp[i-1] = erfinv((i/4) - 1)
        
        self._erf = erf_temp
        self._p = p
        self._alpha_W = p + 2

        self._u0 = u0
        self._alpha_u = alpha_u
        self.target = []
        self.exog = []
        self.test = {}
    #  calculate the sample mean of the data (could be a vector), maybe split into sample mean of each variable
    def sample_mean(self, data):
        # print(sum(data), len(data))
        return sum(np.asarray(data))/len(data)

    
    # calculate the scatter matrix around the mean uN
    # def scatter_matrix(self, data, uN_):
    def scatter_matrix(self, data, uN_):

        # assert data.shape[1] == p * (1 + no_exog_vars), 'Data dimensions do not match expected shape'
        # Assuming data has been preprocessed to include lags of y and X
        temp = data - uN_

        SN = 0
        for row in temp:
            row = row[:, np.newaxis]
            SN += row * row.T
        # print(len(SN),len(SN[0]))
        return SN
    # def WN_func(self, uN_, SN, W0, N):
    def WN_func(self, uN_, SN, W0, N):
        # assert SN.shape[0] == p * (1 + no_exog_vars), 'Scatter matrix dimensions do not match expected shape'
        temp = self._u0 - uN_

        # Assuming scatter matrix has been computed from preprocessed data
        temp = temp[:, np.newaxis]
        WN = W0 + SN + ((self._alpha_u * N) / (self._alpha_u + N)) * np.dot(temp, temp.T)
        return WN
    # Updates the matrix WN_d, calculates the within node covariance matrix 
    def WN_d_func(self, uN_d_, SN_d_, W0_, N_):
        temp = -uN_d_
        temp = temp[:, np.newaxis]
        WN_ = W0_ + SN_d_ + ((self._alpha_u * N_) / (self._alpha_u + N_)) * np.dot(temp, temp.T)
        return WN_
    # calculating the Maximum a posteriori arameters for the ar model
    def MAP_param(self, N, uN_, WN, q):
        ut = ((self._alpha_u * self._u0) + (N * uN_)) / (self._alpha_u + N)
        div = (self._alpha_W + N - (self._p + 1 + q))
        if div == 0:
            div = 0.00001
        Wt_inv = (1 / div) * WN
        return ut, Wt_inv

    def param(self, target, exog):
        # Construct the data matrix (X) using the provided lags
        self.test = {'tar' : target, 'exog' :exog}
        reshaped_data = np.concatenate(exog, axis=1)
        data = np.hstack((target, reshaped_data))
        # Add intercept term (column of ones) to X
        X = np.hstack([np.ones((data.shape[0], 1)), data[:, 1:]])
        y = data[:,0]
        self.target, self.exog = y, data
        coeffs, residuals, rank, s = lstsq(X, y, rcond=None)
        if residuals.size == 0:
            residuals = np.sum((y - X.dot(coeffs))**2)
        var = residuals / len(y)
        m = [coeffs[0]]

        return var, coeffs[1:], m
    # scaling function using the gamma distribution
    def c_func(self, l, alpha):
        c = 1
        #   for loop goes from 1 to l
        for i in range(1, l + 1):
            c *= gamma((alpha + 1 - i) / 2)
        
        return c
    # probability density scaling function used
    def pds_func(self, N, W0, WN):
        pds = (pi**(-((self._p + 1) * N) / 2)) + \
        ((self._alpha_u / (self._alpha_u + N))**((self._p + 1) / 2)) + \
        (self.c_func(self._p + 1, self._alpha_W + N) / self.c_func(self._p + 1, self._alpha_W)) * (det(W0)**(self._alpha_W / 2))*(det(WN)**(-(self._alpha_W + N) / 2))
        return pds
    # similiar to above just now with different params
    def pd_s_func(self, u0_, N_, W0_, WN_):
        pds = (pi**(-((self._p + 1) * N_) / 2)) + \
        ((self._alpha_u / (self._alpha_u + N_))**((self._p + 1) / 2)) + \
        (self.c_func(self._p + 1, self._alpha_W - 1 + N_) / self.c_func(self._p + 1, self._alpha_W - 1)) * (det(W0_)**((self._alpha_W - 1) / 2))*(det(WN_)**(-(self._alpha_W - 1 + N_) / 2))
        return pds
    def mult_func(self, l, alpha, N):
        c = 1
        #   for loop goes from 1 to l
        for i in range(1, l + 1):
            c *= ((alpha + 1 + N - i)/(alpha + 1 - i))
        # print(c)
        return c

    def pds_func2(self, N, W0, WN, u0_, N_, W0_, WN_):

        pds = (det(W0)**(self._alpha_W / 2))*det(WN)**(-(self._alpha_W + N) / 2) / \
        (det(W0_)**((self._alpha_W - 1) / 2))*(det(WN_)**(-(self._alpha_W - 1 + N_) / 2)) * \
        self.mult_func(self._p + 1,self._alpha_W, N)                                                                                                             

        return pds
    def pds_approx(self, N, W0, WN):

        likelihood_val = (self._alpha_u / (self._alpha_u + N))**((self._p + 1) / 2)
        likelihood_val *= (det(W0)**(self._alpha_W / 2))
        likelihood_val *= (det(WN)**(-(self._alpha_W + N) / 2))
        likelihood_val -= self._alpha_W/2 * log(N)
        return likelihood_val
    
    def LeafScore(self, data):
        N = len(data)
        self.target = data
        uN_ = self.sample_mean(data)
        SN = self.scatter_matrix(data, uN_)
        W0 = np.identity(SN.shape[0])
        WN = self.WN_func(uN_, SN, W0, N)
        # ut, Wt_inv = self.MAP_param(N, uN_, WN, 0)
        data_ = data[:-1]
        N_ = len(data_)
        uN_d_ = self.sample_mean(data_)
        SN_d_ = self.scatter_matrix(data_, uN_d_)
        u0_ = 0
        W0_ = inv(inv(W0))
        WN_d_ = self.WN_d_func(uN_d_, SN_d_, W0_, N_)

        if N > 20:
            # pds2 = self.pds_func2(N, W0, WN, u0_, N_, W0_, WN_d_)
            pds = self.pds_approx(N, W0, WN)
            pds_ = self.pds_approx(N_, W0_, WN_d_)
            pds2 = pds/pds_
        else:
            pds = self.pds_func(N, W0, WN)
            pds_ = self.pd_s_func(u0_, N_, W0_, WN_d_)
            pds2 =  pds/pds_
        
        return pds2
        
    # This will spplit a dataset into two froups based on the specific features. Then splits the data points into the left or right set
    def test_split(self, index, value, dataset):
        left, right = list(), list()
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right
    def rest_split(self, index, value, train, exog, ex, var):
                # Initializations
        left, right = [], []
        left_e, right_e = [], []
        
        # Convert to DataFrame for easier operations
        train_df = pd.DataFrame(train)
        exog_dfs = [pd.DataFrame(ex) for ex in exog]

        # Split based on exogenous variable
        if ex == "y":
            # Split the specified variable in exog
            for i, row in exog_dfs[var].iterrows():
                if row[index] < value:
                    left_e.append(i)
                else:
                    right_e.append(i)
            if not left_e or not right_e:
                return None, None, None, None
            left_e_index = pd.Index(left_e)
            right_e_index = pd.Index(right_e)

            # Extract corresponding rows from other variables in exog and from train
            left_exog = [df.loc[left_e_index] for df in exog_dfs]
            right_exog = [df.loc[right_e_index] for df in exog_dfs]

            left = train_df.loc[left_e_index]
            right = train_df.loc[right_e_index]

        # Split based on training data
        else:
            for i, row in train_df.iterrows():
                if row[index] < value:
                    left.append(i)
                else:
                    right.append(i)
            if not left or not right:
                return None, None, None, None
            
            left_index = pd.Index(left)
            right_index = pd.Index(right)

            left = train_df.loc[left_index]
            right = train_df.loc[right_index]

            left_exog = [df.loc[left_index] for df in exog_dfs]
            right_exog = [df.loc[right_index] for df in exog_dfs]
        left = left.values.tolist()
        right = right.values.tolist()
        left_exog = [df.values.tolist() for df in left_exog]
        right_exog = [df.values.tolist() for df in right_exog]
        return left, right, left_exog, right_exog
    # itrates through the features and the values to det the best for splitting the dataset, calkculates the score for each split and choses the one with best improvement
    def get_split(self, train, train_exog):
        self.target, self.exog = train, train_exog
        b_index, b_value, b_groups, var = 999, 999, None, 'y'
        b_score = self.LeafScore(train)
        avg = np.mean(train, axis=0)[:-1]
        sigma = np.std(train, axis=0)[:-1]
        split_data = train, train_exog
        for index in range(len(avg)):
            for i in range(len(self._erf)):

                value = avg[index] + sigma[index] * self._erf[i]
                # groups = self.test_split(index, value, train)
                data = self.rest_split(index, value, train, train_exog,"n", 0)
                groups = data[0], data[1]
                if data is None or data[0] is None or data[1] is None:
                    # print("NONE TYPE")
                    continue
                new_score = 1
                for group in groups:
                    if len(group) > 1:
                        new_score *= self.LeafScore(group)
            
                        if new_score > b_score:
                            b_index, b_value, b_score, b_groups, var, split_data = index, value, new_score, groups, ("y"+str(index)), data
        j = 0
        bb_score = 1
        for ex in train_exog:
            # print(j)
            self.exog = ex
            avg = np.mean(ex, axis=0)
            min_e = np.min(ex, axis=0)
            max_e = np.max(ex, axis=0)
            self.target = avg
            bb_score =  max(bb_score, self.LeafScore(ex))
            # print(avg)
            sigma = np.std(ex, axis=0)
            for index in range(len(avg)):
                for i in range(len(self._erf)):
                    value = avg[index] + sigma[index] * self._erf[i]
                    if value <= min_e[index] or value >= max_e[index]:
                        continue
                    # groups = self.test_split(index, value, ex)
                    data = self.rest_split(index, value, train, train_exog,"y", j)
                    if data is None or data[2][j] is None or data[3][j] is None:
                        # print("NONE TYPE")
                        continue
                    self.target = train
                    self.exog = train_exog
                    groups = data[2][j], data[3][j]
                    
                    new_score = 1
                    for group in groups:
                        if len(group) >1:
                            new_score *= self.LeafScore(group)
                
                            if new_score > b_score:
                                b_index, b_value, b_score, b_groups, var, split_data = index, value, new_score, groups, ("x"+str(j)+str(index)), data
            j += 1
        # print({'index':b_index, 'variable': var, 'value':b_value, 'groups':b_groups})
        # print(var, b_score, bb_score)
        # print(np.asarray(data).shape)
        # self.exog = exog
        return {'index':b_index, 'variable': var, 'value':b_value, 'groups':b_groups, 'split_data':split_data}
    # turns a group of points, belonging to one datagroup into a terminal node, calculates the parameters for that specific group
    def to_terminal(self, target, exog):
        outcomes = self.param(target,exog)
        return outcomes
    # this recursivelky builds the tree up. If the node should be a terminal node make terminal, else use get split to find next best split
    def split(self, node, max_depth, min_size, depth):
        left_t, right_t, left_e, right_e = node['split_data']
        left, right = node['groups']

        del(node['groups'])
        if not left or not right:
            node['left'] = node['right'] = self.to_terminal(left_t + right_t, left_e + right_e)
            return
        
        if depth >= max_depth:
            node['left'], node['right'] = self.to_terminal(left_t,left_e ), self.to_terminal(right_t, right_e)
            return
        
        if len(left) <= min_size:
            node['left'] = self.to_terminal(left_t, left_e)
        else:
            node['left'] = self.get_split(left_t, left_e)
            if node['left']['groups'] is None:
                node['left'] = self.to_terminal(left_t, left_e)
            else:
                self.split(node['left'], max_depth, min_size, depth+1)
        
        if len(right) <= min_size:
            node['right'] = self.to_terminal(right_t, right_e)
        else:
            node['right'] = self.get_split(right_t, right_e)
            if node['right']['groups'] is None:
                node['right'] = self.to_terminal(right_t, right_e)
            else:
                self.split(node['right'], max_depth, min_size, depth+1)
    # initiates the buiilding process. Finds initial split and if there are no effective splits then makes source node a terminal node
    def build_tree(self, train, train_exog, max_depth, min_size):
        train=train
        train_exog = train_exog
        root = self.get_split(train, train_exog)
        if root['groups'] is None:
            root['root'] = self.to_terminal(train,train_exog)
            root['index'] = None
            root['value'] = None
            del(root['groups'])
        else:
            # print(root['index'])
            self.split(root, max_depth, min_size, 1)
        
        return root
    # prints the tree structure 
    def print_tree(self, node, depth=0):
        if isinstance(node, dict):
            if node['value'] is None:
                print(node)
                return                                                                                                                               
            print('%s[%s < %.3f]' % ((depth*' ', (node['variable']), node['value'])))
            # print(depth)
            # print(node)
            self.print_tree(node['left'], depth+1)
            self.print_tree(node['right'], depth+1)
    
        else:
            output = """{} [var: {}
        {} parameters: {}
        {} m: {}]""".format(depth*' ', np.round(node[0],2), depth*' '," ".join(map(str,np.round(np.array(node[1][:]),4).flatten())), depth*' ',  np.round(node[2][0],2))
            print('%s%s' % ((depth*' ', output)))
    # follows the tree strarting from root node until a terminal node is found
    def predict(self, node, row):
        # If the node is a terminal node, return its value
        if 'root' in node:
            return node['root']

        # Extract the variable type (either 'y' or 'x') and its index
        var_type = node['variable'][0]
        
        # If it's the target variable (y), the index is directly the second part of the variable key
        if var_type == 'y':
            var_index = int(node['variable'][1:])
        
        # If it's an exogenous variable (x), adjust the index
        else:
            exog_num = int(node['variable'][1])  # which exogenous variable (e.g., x1, x2, etc.)
            exog_lag = int(node['variable'][2:])  # lag of the exogenous variable
            var_index = self._p + (exog_num - 1) * self._p + exog_lag

        # Navigate the tree based on the comparison at the current node
        if row[var_index] < node['value']:
            if isinstance(node['left'], dict):
                return self.predict(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predict(node['right'], row)
            else:
                return node['right']

def hit_rate(ts_true, ts_pred):
    diff_true = np.diff(ts_true)
    diff_pred = np.diff(ts_pred)
    return np.sum(np.sign(diff_true) == np.sign(diff_pred)) / len(diff_true)

def time_series_pred(data, max_p, preprocessing_method, max_depth, min_size):
    ts_train, ts_valid, ts_test, ts_param = preprocessing(data, method=preprocessing_method)
    p_set = [i for i in range(1,max_p+1)]
    valid_prediction_list = []
    valid_prediction_list_cumsum = []
    #   Example
    idx = 0
    d_val = np.array(ts_valid)[0]
    max_len = len(d_val) - max_p
    d_test = ts_test[0]
    tree_list = []
    rmse_ART_list = []
    hit_rate_ART_list = []

    for p in p_set:
            print("tree model with {} lags".format(p))
            train=[]
            valid=[]
            for ind in range(len(ts_train)):
                s, l = ts_train[ind], ts_valid[ind]
                full = np.append(s,l)
                temp = []
                temp_valid = []
                for i in range(len(full)-(p+1)):
                    if i < len(s)-(p+1):
                        temp.append(full[i:i + p + 1])
                    else:
                        temp_valid.append(full[i:i + p + 1])
                train.append(temp)
                valid.append(temp_valid)
            d = train[0]
            d_exog = train[1:]

            comb = np.concatenate(train, axis=1)
            comb_val = np.concatenate(valid, axis=1)

            ART = AutoregressiveTree(p)
            tree = ART.build_tree(d, d_exog, max_depth, min_size)
            tree_list.append(tree)

            valid_prediction = []
            valid_window = comb_val[p-1][1:]

            for i in range(len(comb_val)):
                if i >= p:
                    parameters = ART.predict(tree, valid_window)
                    prediction_temp = np.dot(valid_window[:,np.newaxis].T,parameters[1]) + parameters[2]
                    valid_prediction.append(prediction_temp[0])
                valid_window = comb_val[i][1:]
            if preprocessing_method is 'differencing':
                train_s = pd.Series(ts_train[idx], copy=True).cumsum()
                last_value_train= pd.Series.tolist(train_s)[-1]
                valid_prediction_temp = [0]*(len(valid_prediction)+1)
                valid_prediction_temp[1:] = valid_prediction
                valid_prediction_temp[0] = last_value_train
                valid_prediction_temp = pd.Series(valid_prediction_temp, copy=True)
                valid_prediction_cumsum = valid_prediction_temp.cumsum()
                valid_prediction_list_cumsum.append(valid_prediction_cumsum)
                        
                
            if preprocessing_method is 'normalization':
                valid_prediction_list.append(valid_prediction[:max_len])
                valid_prediction = pd.Series(valid_prediction[:max_len], copy=True)
                d_val_mean = ts_param[idx][2]
                d_val_std = ts_param[idx][3]
                valid_prediction_denorm= (valid_prediction * d_val_std) + d_val_mean
                valid_prediction_list_cumsum.append(valid_prediction_denorm)


    if preprocessing_method is 'differencing':
        d_val_cumsum = np.array(ts_valid[idx]).cumsum()
    elif preprocessing_method is 'normalization':
        d_val_mean = ts_param[idx][2]
        d_val_std = ts_param[idx][3]
        d_val_cumsum = (d_val[:max_len] * d_val_std) + d_val_mean
    else:
        d_val_cumsum = d_val[:max_len]
    
    for i in max_p:
        hit_rate_ART_list.append(hit_rate(d_val_cumsum, valid_prediction_list_cumsum[i]))

        rmse_ART_list.append(sqrt(mean_squared_error(d_val_cumsum, valid_prediction_list_cumsum[i])))

        
    return d_val_cumsum, valid_prediction_list_cumsum, tree_list, hit_rate_ART_list, rmse_ART_list

