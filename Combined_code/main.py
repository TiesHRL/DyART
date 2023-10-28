import ARXT
from ARXT import hit_rate
import Data_gen
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
import matplotlib.pyplot as plt
import logging

def get_data():

    tickers = ["^GSPC", "^IXIC", "^DJI","JPYUSD=X", "^VIX"] # , "GBPUSD=X", "EURUSD=X",
    data = Data_gen.collect_data(tickers)
    return(data)

def train_run_tree(data, p, max_depth, min_size):
    p, max_depth, min_size =  int(round(p)), int(round(max_depth)), int(round(min_size))
    d_val_cumsum, valid_prediction_cumsum, tree_list = ARXT.time_series_pred(data=data, p=p, preprocessing_method='normalization', max_depth=max_depth, min_size=min_size)
    hit_rate_sample = hit_rate(d_val_cumsum, valid_prediction_cumsum)

    rmse_sample = (sqrt(mean_squared_error(d_val_cumsum, valid_prediction_cumsum)))

    return d_val_cumsum, valid_prediction_cumsum, tree_list, hit_rate_sample, rmse_sample
DATA = get_data()

def objective_function(p, max_depth, min_size):
    # Set up and train the ART model using the hyperparameters
    p, max_depth, min_size =  round(p), round(max_depth), round(min_size)
    d_val_cumsum, valid_prediction_cumsum, _, hit_rate_sample, rmse_sample = train_run_tree(DATA[0:1000], p, max_depth, min_size)

    performance = hit_rate_sample * 2 - rmse_sample * 0.5
    return performance

def main():
    # Define hyperparameter bounds
    pbounds = {
        "p": (5, 15),
        "max_depth":(20, 25),
        "min_size":(20, 30)
    }

    # Create BayesianOptimization object
    optimizer = BayesianOptimization(f=objective_function, pbounds=pbounds, random_state=1)
    acq_function = UtilityFunction(kind="ei", kappa=5, kappa_decay=0.8)
    # Run optimization
    # optimizer.maximize(init_points=2, n_iter=5, acquisition_function = acq_function)
    
    # Print best results
    # print(optimizer.max)
    # opt_params  = optimizer.max['params']
    opt_params = {'max_depth': 21.463686043158983, 'min_size': 21.523643262828223, 'p': 5.902789263724006}

    d_val_cumsum, valid_prediction_cumsum, tree, _, _ = train_run_tree(data=DATA[300:1000], p=opt_params['p'], max_depth=opt_params['max_depth'], min_size=opt_params['min_size'])
    
    print(opt_params['p'])
    ART = ARXT.AutoregressiveTree(opt_params['p'])
    ART.print_tree(tree)
    plt.plot(d_val_cumsum, label = "real")
    plt.plot(valid_prediction_cumsum, label= 'pred')
    plt.show()
    return optimizer.max


if __name__ == "__main__":
    main()