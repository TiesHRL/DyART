import ARXT
from ARXT import hit_rate
import Data_gen
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
import matplotlib.pyplot as plt
import logging
import bayesian_changepoint_detection.bayesian_models as bayes_models
from bayesian_changepoint_detection.hazard_functions import constant_hazard
import bayesian_changepoint_detection.online_likelihoods as online_ll
import json 

def get_data():

    # tickers = ["^GSPC", "^IXIC", "^DJI","JPYUSD=X", "^VIX"] # , "GBPUSD=X", "EURUSD=X",
    # data = Data_gen.collect_data(tickers)
    data = pd.read_csv("Data/fin_data.csv")
    # data.to_csv("Data/fin_data.csv")
    data.index = data["Date"]
    data = data.drop(["Date"], axis=1)
    data.columns = range(data.shape[1])

    return(data)

def train_run_tree(data, p, max_depth, min_size, splt):
    p, max_depth, min_size =  int(round(p)), int(round(max_depth)), int(round(min_size))
    d_val_cumsum, valid_prediction_cumsum, tree_list = ARXT.ARXT_time_series_pred(data=data, p=p, preprocessing_method='normalization', max_depth=max_depth, min_size=min_size, splt=splt)
    hit_rate_sample = hit_rate(d_val_cumsum, valid_prediction_cumsum)

    rmse_sample = (sqrt(mean_squared_error(d_val_cumsum, valid_prediction_cumsum)))

    return d_val_cumsum, valid_prediction_cumsum, tree_list, hit_rate_sample, rmse_sample
DATA = get_data()

def objective_function(p, max_depth, min_size, start, fin, splt):
    # Set up and train the ART model using the hyperparameters
    p, max_depth, min_size =  round(p), round(max_depth), round(min_size)
    d_val_cumsum, valid_prediction_cumsum, _, hit_rate_sample, rmse_sample = train_run_tree(DATA[start:fin], p, max_depth, min_size, splt=splt)

    performance = hit_rate_sample * 2 - rmse_sample * 0.5
    return performance
def optimizer(pbounds, start, fin, splt):
    optimizer = BayesianOptimization(f= lambda p, max_depth, min_size: objective_function(p, max_depth, min_size,start, fin, splt), pbounds=pbounds, random_state=1)
    acq_function = UtilityFunction(kind="ei", kappa=5, kappa_decay=0.8)
    optimizer.maximize(init_points=2, n_iter=5, acquisition_function = acq_function)
    opt_params  = optimizer.max['params']
    return opt_params
def ARXT_tree(splt):
    train_len = 1000
    # Define hyperparameter bounds
    pbounds = {
        "p": (5, 20),
        "max_depth":(10, 25),
        "min_size":(10, 30)
    }
    opt_params = optimizer(pbounds, 0 , train_len, splt)
    opt_params = {"max_depth": 21.46, "min_size": 21.52, "p": 5.903}

    p, max_depth, min_size = round(opt_params['p']), round(opt_params['max_depth']), round(opt_params['min_size'])
    next_pbounds = {"p": (p*0.7, p*1.3), "max_depth" : (max_depth*0.7, max_depth*1.3), "min_size" : (min_size*0.7, min_size*1.3)}

    ART = ARXT.AutoregressiveTree(p, splt=splt)    

    _, _, tree, _, _ = train_run_tree(data=DATA.iloc[200:train_len], p=p, max_depth=max_depth, min_size=min_size, splt=splt)
    c_det = bayes_models.OnlineChagepoint(np.array(DATA[0]), constant_hazard, 200)
    log_likelihood_class = c_det.warm_run(llc = online_ll.StudentT(alpha=0.1, beta=.01, kappa=1, mu=0),t = train_len)
    Nw = 200
    forecasts = []
    
    for i in range(train_len, len(DATA[0])):
        forecasts.append(ARXT.forecast(DATA.iloc[i-200:i], tree, ART, p))

        if c_det.iteration(i, log_likelihood_class, Nw):
            print("retraining at ", DATA.index[i])
            opt_params = optimizer(next_pbounds, i-500, i, splt)
            p, max_depth, min_size = round(opt_params['p']), round(opt_params['max_depth']), round(opt_params['min_size'])
            next_pbounds = {"p": (p*0.7, p*1.3), "max_depth" : (max_depth*0.7, max_depth*1.3), "min_size" : (min_size*0.7, min_size*1.3)}

            ART = ARXT.AutoregressiveTree(p, splt=splt)    
    
            _, _, tree, _, _ = train_run_tree(data=DATA[i-600:i], p=p, max_depth=max_depth, min_size=min_size, splt=splt)

    pd.DataFrame(forecasts).to_csv("forecasts_{}.csv".format(splt))

    plt.plot(DATA.iloc[train_len:,0], label="truth")
    plt.plot(forecasts, label="forecasts")
    plt.legend()
    plt.show()

def AR_model(p):
    train_len = 1000
    # Define hyperparameter bounds
    pbounds = {
        "p": (5, 20),
        "max_depth":(10, 25),
        "min_size":(10, 30)
    }
    # opt_params = optimizer(pbounds, 0 , train_len, "target")
    # opt_params = {"max_depth": 21.46, "min_size": 21.52, "p": 5.903}

    # p, max_depth, min_size = round(opt_params['p']), round(opt_params['max_depth']), round(opt_params['min_size'])
    # next_pbounds = {"p": (p*0.7, p*1.3), "max_depth" : (max_depth*0.7, max_depth*1.3), "min_size" : (min_size*0.7, min_size*1.3)}

    AR = ARXT.AR_p(DATA, p)    

    AR.AR_p_model(train_len)
    forecasts = []
    for i in range(train_len, len(DATA[0])):
        forecasts.append(AR.predict(DATA[i-p:i]))

        # if c_det.iteration(i, log_likelihood_class, Nw):
        #     print("retraining at ", DATA.index[i])
        #     opt_params = optimizer(next_pbounds, i-500, i, splt)
        #     p, max_depth, min_size = round(opt_params['p']), round(opt_params['max_depth']), round(opt_params['min_size'])
        #     next_pbounds = {"p": (p*0.7, p*1.3), "max_depth" : (max_depth*0.7, max_depth*1.3), "min_size" : (min_size*0.7, min_size*1.3)}

        #     ART = ARXT.AutoregressiveTree(p, splt=splt)    
    
        #     _, _, tree, _, _ = train_run_tree(data=DATA[i-600:i], p=p, max_depth=max_depth, min_size=min_size, splt=splt)

    pd.DataFrame(forecasts).to_csv("forecasts_AR.csv")

    plt.plot(DATA.iloc[train_len:,0], label="truth")
    plt.plot(forecasts, label="forecasts")
    plt.legend()
    plt.show()
def main():
    AR_model(5)

if __name__ == "__main__":
    main()