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
import time 

def get_data():

    # tickers = ["^GSPC", "^IXIC", "^DJI","JPYUSD=X", "^VIX", "GBPUSD=X", "EURUSD=X] # , "GBPUSD=X", "EURUSD=X",
    # data = Data_gen.collect_data(tickers)
    data = pd.read_csv("Data/fin_data.csv")
    # data.to_csv("Data/fin_data.csv")
    data.index = data["Date"]
    data = data.drop(["Date"], axis=1)
    data.columns = range(data.shape[1])

    return(data)

def train_run_tree(data, p, max_depth, min_size, max_weight, splt):
    p, max_depth, min_size =  int(round(p)), int(round(max_depth)), int(round(min_size))
    d_val_cumsum, valid_prediction_cumsum, tree_list = ARXT.ARXT_time_series_pred(data=data, p=p, preprocessing_method='normalization', max_depth=max_depth, min_size=min_size, max_weight=max_weight, splt=splt)
    hit_rate_sample = hit_rate(d_val_cumsum, valid_prediction_cumsum)

    rmse_sample = (sqrt(mean_squared_error(d_val_cumsum, valid_prediction_cumsum)))

    return d_val_cumsum, valid_prediction_cumsum, tree_list, hit_rate_sample, rmse_sample
DATA = get_data()

def objective_function(p, max_depth, min_size, max_weight, start, fin, splt):
    # Set up and train the ART model using the hyperparameters
    p, max_depth, min_size, max_weight =  round(p), round(max_depth), round(min_size), round(max_weight)
    d_val_cumsum, valid_prediction_cumsum, _, hit_rate_sample, rmse_sample = train_run_tree(DATA[start:fin], p, max_depth, min_size, max_weight, splt=splt)

    performance = hit_rate_sample * 2 - rmse_sample * 0.5
    return performance
def optimizer(pbounds, start, fin, splt, init_points=10, n_iter=30):
    optimizer = BayesianOptimization(f= lambda p, max_depth, min_size, max_weight: objective_function(p, max_depth, min_size, max_weight, start, fin, splt), pbounds=pbounds, random_state=1)
    acq_function = UtilityFunction(kind="ei", kappa=5, kappa_decay=0.8)
    optimizer.maximize(init_points, n_iter, acquisition_function = acq_function)
    opt_params  = optimizer.max['params']
    return opt_params
def ARXT_tree(splt, tune):
    start_time = time.time()
    train_len = 1000
    # Define hyperparameter bounds
    pbounds = {
        "p": (5, 20),
        "max_depth":(10, 150),
        "min_size":(1, 50),
        "max_weight": (0.01, 0.15)
    }
    opt_params = optimizer(pbounds, 0 , train_len, splt)
    p, max_depth, min_size, max_weight = round(opt_params['p']), round(opt_params['max_depth']), round(opt_params['min_size']), opt_params['max_weight']
    next_pbounds = {"p": (p*0.7, p*1.3), "max_depth" : (max_depth*0.7, max_depth*1.3), "min_size" : (min_size*0.7, min_size*1.3), "max_weight" : (max(0.001, max_weight*0.7), max_weight*1.3)}

    ART = ARXT.AutoregressiveTree(p, splt=splt)    

    _, _, tree, _, _ = train_run_tree(data=DATA.iloc[200:train_len], p=p, max_depth=max_depth, min_size=min_size, max_weight=max_weight, splt=splt)
    c_det = bayes_models.OnlineChagepoint(np.array(DATA[0]), constant_hazard, 200)
    log_likelihood_class = c_det.warm_run(llc = online_ll.StudentT(alpha=0.1, beta=.01, kappa=1, mu=0),t = train_len)
    Nw = 200
    forecasts = []
    retraining_points = []
    for i in range(train_len, len(DATA[0])):
        forecasts.append(ARXT.forecast(DATA.iloc[i-200:i], tree, ART, p))

        if c_det.iteration(i, log_likelihood_class, Nw):
            print("retraining at ", DATA.index[i])
            retraining_points.append(DATA.index[i])
            if tune:
                opt_params = optimizer(next_pbounds, i-500, i, splt, init_points=5, n_iter = 10)
                p, max_depth, min_size, max_weight = round(opt_params['p']), round(opt_params['max_depth']), round(opt_params['min_size']), opt_params['max_weight']
                next_pbounds = {"p": (p*0.7, p*1.3), "max_depth" : (max_depth*0.7, max_depth*1.3), "min_size" : (min_size*0.7, min_size*1.3), "max_weight" : (max(0.001, max_weight*0.7), max_weight*1.3)}

            ART = ARXT.AutoregressiveTree(p, splt=splt)    
    
            _, _, tree, _, _ = train_run_tree(data=DATA[i-600:i], p=p, max_depth=max_depth, min_size=min_size, max_weight=max_weight, splt=splt)
    retrain = "retrain"
    if tune: retrain = "retune"
    pd.DataFrame(forecasts).to_csv("forecasts_{}_{}.csv".format(splt, retrain))
    # pd.DataFrame(retraining_points).to_csv("retraining_points.csv".format(splt))

    # plt.plot(DATA.iloc[train_len:,0], label="truth")
    # plt.plot(forecasts, label="forecasts")
    # plt.legend()
    # plt.show()
    end_time = time.time()
    duration = end_time-start_time
    print("Time taken for ARXT {} {}: {}".format(splt, retrain, duration))
    return forecasts, retraining_points

def AR_model(p, train):
    start_time = time.time()
    train_len = 1000
    AR = ARXT.AR_p(DATA, p)    

    AR.AR_p_model(0, train_len)
    c_det = bayes_models.OnlineChagepoint(np.array(DATA[0]), constant_hazard, 200)
    log_likelihood_class = c_det.warm_run(llc = online_ll.StudentT(alpha=0.1, beta=.01, kappa=1, mu=0),t = train_len)
    Nw = 200
    forecasts = []
    retraining_points = []
    for i in range(train_len, len(DATA[0])):
        if train:
            if c_det.iteration(i, log_likelihood_class, Nw):
                print("retraining at ", DATA.index[i])
                retraining_points.append(DATA.index[i])
                AR.AR_p_model(max(0,i-500), i)

        forecasts.append(AR.predict(DATA[i-p:i]))

    # plt.plot(DATA.iloc[train_len:,0], label="truth")
    # plt.plot(forecasts, label="forecasts")
    # plt.legend()
    # plt.show()
    retrain = ""
    if train: retrain = "retune"
    end_time = time.time()
    duration = end_time-start_time
    print("Time taken for AR(p) {}: {}".format(retrain, duration))
    pd.DataFrame(forecasts).to_csv("forecasts_AR_{}.csv".format(retrain))

    return forecasts
def main():
    ARTX_exog_tuned, retraining_points = ARXT_tree("exog", True)
    ARTX_exog_trained, _ = ARXT_tree("exog", False)
    ARTX_target_tuned, _ = ARXT_tree("target", True)
    ARTX_target_trained, _ = ARXT_tree("target", False)
    AR_p_trained =  AR_model(5, True)
    AR_p =  AR_model(5, False)
    pd.DataFrame(retraining_points).to_csv("Data\\retraining_points.csv")
    for i, forecast in enumerate([ARTX_exog_tuned, ARTX_exog_trained, ARTX_target_tuned, ARTX_target_trained, AR_p_trained, AR_p]):
        pd.DataFrame(forecast).to_csv("Data\\results_{}.csv".format(i))
    pd.DataFrame([ARTX_exog_tuned, ARTX_exog_trained, ARTX_target_tuned, ARTX_target_trained, AR_p_trained, AR_p]).to_csv("Data\\results.csv")
if __name__ == "__main__":
    main()