from networkx import difference
import ARXT
from ARXT import hit_rate
import data_gen
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from math import sqrt
from bayes_opt import BayesianOptimization, UtilityFunction
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
import matplotlib.pyplot as plt
# import bayesian_changepoint_detection.bayesian_models as bayes_models
# from bayesian_changepoint_detection.hazard_functions import constant_hazard
# import bayesian_changepoint_detection.online_likelihoods as online_ll
from statsmodels.tsa.arima.model import ARIMA
import time 

def get_data(differencing = False):

    # tickers = ["^GSPC", "^IXIC", "^DJI","JPYUSD=X", "^VIX", "GBPUSD=X", "EURUSD=X] # , "GBPUSD=X", "EURUSD=X",
    # data = data_gen.collect_data(tickers)
    data = pd.read_csv("Data/fin_data.csv")
    # data.to_csv("data/fin_data.csv")
    data.index = data["Date"]
    data = data.drop(["Date"], axis=1)
    data = data.pct_change()
    data = data.iloc[1:,:]

    data.columns = range(data.shape[1])
    if differencing:
        data = data.diff()
        data = data.iloc[1:,:]

    else:
        data = pd.DataFrame(np.log1p(data))
    data = data*100

    return(data)

def train_run_tree(data, p, max_depth, min_size, max_weight, splt):
    p, max_depth, min_size =  int(round(p)), int(round(max_depth)), int(round(min_size))
    d_val_cumsum, valid_prediction_cumsum, tree_list = ARXT.ARXT_time_series_pred(data=data, p=p, preprocessing_method='normalization', max_depth=max_depth, min_size=min_size, max_weight=max_weight, splt=splt)
    hit_rate_sample = hit_rate(d_val_cumsum, valid_prediction_cumsum)

    rmse_sample = (mean_absolute_percentage_error(d_val_cumsum, valid_prediction_cumsum))

    return d_val_cumsum, valid_prediction_cumsum, tree_list, hit_rate_sample, rmse_sample

def train_run_ART(data, p, max_depth, min_size):
    p, max_depth, min_size =  int(round(p)), int(round(max_depth)), int(round(min_size))
    d_val_cumsum, valid_prediction_cumsum, tree_list = ARXT.ART_time_series_pred(data=data, p=p, preprocessing_method='normalization', max_depth=max_depth, min_size=min_size)
    hit_rate_sample = hit_rate(d_val_cumsum, valid_prediction_cumsum)

    rmse_sample = (mean_absolute_percentage_error(d_val_cumsum, valid_prediction_cumsum))

    return d_val_cumsum, valid_prediction_cumsum, tree_list, hit_rate_sample, rmse_sample
# define and load in to global variables 
CPS = pd.read_csv('data/changepoints.csv').to_numpy()

def objective_function(p, max_depth, min_size, max_weight, start, fin, splt, data, ART_bool):
    # Set up and train the ART model using the hyperparameters
    if ART_bool:
        p, max_depth, min_size =  round(p), round(max_depth), round(min_size)
        d_val_cumsum, valid_prediction_cumsum, _, hit_rate_sample, rmse_sample = train_run_ART(data[start:fin], p, max_depth, min_size)

    else:
        p, max_depth, min_size, max_weight =  round(p), round(max_depth), round(min_size), max_weight
        d_val_cumsum, valid_prediction_cumsum, _, hit_rate_sample, rmse_sample = train_run_tree(data[start:fin], p, max_depth, min_size, max_weight, splt=splt)

    performance = 2 * hit_rate_sample - rmse_sample * 0.5
    return performance

def optimizer(pbounds, start, fin, splt, retune_it, data, ART_bool, init_points=10, n_iter=30):
    if ART_bool:
        optimizer = BayesianOptimization(f= lambda p, max_depth, min_size: objective_function(p, max_depth, min_size, 0, start, fin, splt, data, ART_bool), pbounds=pbounds, random_state=1)
    else:
        optimizer = BayesianOptimization(f= lambda p, max_depth, min_size, max_weight: objective_function(p, max_depth, min_size, max_weight, start, fin, splt, data, ART_bool), pbounds=pbounds, random_state=1)
    acq_function = UtilityFunction(kind="ei", kappa=5, kappa_decay=0.8)
    # logger = JSONLogger(path=f"./logs_{splt}_{retune_it}.log")
    # optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    optimizer.maximize(init_points, n_iter, acquisition_function = acq_function)
    opt_params  = optimizer.max['params']
    return opt_params

def ART_tree(tune, data):
    ART_bool = True
    splt = False
    start_time = time.time()
    train_len = 1000
    # Define hyperparameter bounds
    retrain = "retrain"
    if tune: retrain = "retune"
    pbounds = {
        "p": (1, 20),
        "max_depth":(10, 150),
        "min_size":(1, 50)
        }

    opt_params = optimizer(pbounds, 0 , train_len, splt, 0, data, ART_bool)
    p, max_depth, min_size = round(opt_params['p']), round(opt_params['max_depth']), round(opt_params['min_size'])
    next_pbounds = {"p": (p*0.7, p*1.3), "max_depth" : (max_depth*0.7, max_depth*1.3), "min_size" : (min_size*0.7, min_size*1.3)}

    ART = ARXT.AutoregressiveTree(p)    

    _, _, tree, _, _ = train_run_ART(data=data.iloc[200:train_len], p=p, max_depth=max_depth, min_size=min_size)
    # c_det = bayes_models.OnlineChagepoint(np.array(data[0]), constant_hazard, 200)
    # log_likelihood_class = c_det.warm_run(llc = online_ll.StudentT(alpha=0.1, beta=.01, kappa=1, mu=0),t = train_len)
    # Nw = 200
    forecasts = []
    retuning = 1
    for i in range(train_len, len(data[0])):
        forecasts.append(ARXT.forecast_ART(data.iloc[i-200:i], tree, ART, p))
        if tune:
            if data.index[i] in CPS:
                print("retraining at ", data.index[i])
                opt_params = optimizer(next_pbounds, i-500, i, splt, retuning, data, ART_bool, init_points=5, n_iter = 10)
                retuning += 1
                p, max_depth, min_size, = round(opt_params['p']), round(opt_params['max_depth']), round(opt_params['min_size'])
                next_pbounds = {"p": (p*0.7, p*1.3), "max_depth" : (max_depth*0.7, max_depth*1.3), "min_size" : (min_size*0.7, min_size*1.3)}

                ART = ARXT.AutoregressiveTree(p)    
        
                _, _, tree, _, _ = train_run_ART(data=data[i-600:i], p=p, max_depth=max_depth, min_size=min_size)
    end_time = time.time()
    duration = end_time-start_time
    print("Time taken for ART {} : {} mins".format(retrain, round(duration)/60))
    return forecasts
def ARXT_tree(splt, tune, data):
    start_time = time.time()
    train_len = 1000
    ART_bool = False
    # Define hyperparameter bounds
    retrain = "retrain"
    if tune: retrain = "retune"
    pbounds = {
        "p": (1, 20),
        "max_depth":(10, 150),
        "min_size":(1, 50),
        "max_weight": (0.01, 0.15)
    }

    opt_params = optimizer(pbounds, 0 , train_len, splt, 0, data, ART_bool)
    p, max_depth, min_size, max_weight = round(opt_params['p']), round(opt_params['max_depth']), round(opt_params['min_size']), opt_params['max_weight']
    next_pbounds = {"p": (p*0.7, p*1.3), "max_depth" : (max_depth*0.7, max_depth*1.3), "min_size" : (min_size*0.7, min_size*1.3), "max_weight" : (max(0.001, max_weight*0.7), max_weight*1.3)}

    ART = ARXT.AutoregressiveXTree(p, splt=splt)    

    _, _, tree, _, _ = train_run_tree(data=data.iloc[200:train_len], p=p, max_depth=max_depth, min_size=min_size, max_weight=max_weight, splt=splt)
    # c_det = bayes_models.OnlineChagepoint(np.array(data[0]), constant_hazard, 200)
    # log_likelihood_class = c_det.warm_run(llc = online_ll.StudentT(alpha=0.1, beta=.01, kappa=1, mu=0),t = train_len)
    # Nw = 200
    forecasts = []
    retuning = 1
    for i in range(train_len, len(data[0])):
        forecasts.append(ARXT.forecast(data.iloc[i-200:i], tree, ART, p))
        if tune:
            if data.index[i] in CPS:
                print("retraining at ", data.index[i])
                opt_params = optimizer(next_pbounds, i-500, i, splt, retuning, data, ART_bool, init_points=5, n_iter = 10)
                retuning += 1
                p, max_depth, min_size, max_weight = round(opt_params['p']), round(opt_params['max_depth']), round(opt_params['min_size']), opt_params['max_weight']
                next_pbounds = {"p": (p*0.7, p*1.3), "max_depth" : (max_depth*0.7, max_depth*1.3), "min_size" : (min_size*0.7, min_size*1.3), "max_weight" : (max(0.001, max_weight*0.7), max_weight*1.3)}

                ART = ARXT.AutoregressiveXTree(p, splt=splt)    
        
                _, _, tree, _, _ = train_run_tree(data=data[i-600:i], p=p, max_depth=max_depth, min_size=min_size, max_weight=max_weight, splt=splt)
    end_time = time.time()
    duration = end_time-start_time
    print("Time taken for ARXT {} {}: {} mins".format(splt, retrain, round(duration)/60))
    return forecasts

def ARX_model(p, train, data):
    start_time = time.time()
    train_len = 1000
    AR = ARXT.ARX_p(data, p)    

    AR.ARX_p_model(0, train_len)
    # c_det = bayes_models.OnlineChagepoint(np.array(data[0]), constant_hazard, 200)
    # log_likelihood_class = c_det.warm_run(llc = online_ll.StudentT(alpha=0.1, beta=.01, kappa=1, mu=0),t = train_len)
    forecasts = []
    for i in range(train_len, len(data[0])):
        if train:
            if data.index[i] in CPS:
                print("retraining at ", data.index[i])
                AR.ARX_p_model(max(0,i-500), i)

        forecasts.append(AR.predict(data[i-p:i]))
    retrain = ""
    if train: retrain = "retune"
    end_time = time.time()
    duration = end_time-start_time
    print("Time taken for ARX(p) {}: {} mins".format(retrain, round(duration)/60))
    # pd.dataFrame(forecasts).to_csv("forecasts_AR_{}.csv".format(retrain))

    return forecasts
def AR_model(p, data):
    start_time = time.time()
    train_len = 1000
    forecasts = []
    for i in range(train_len, len(data[0])):
        moving_val = data.iloc[i-p*5:i,0]
        model = ARIMA(moving_val, order=(p,1,0))
        model_fit = model.fit()
        forecasts.append(model_fit.forecast()[0])
    end_time = time.time()
    duration = end_time-start_time
    print("Time taken for AR({}): {} mins".format(p, duration/60))

    return forecasts
def main():
    differencing = True
    data = get_data(differencing=differencing)
    print(data)

    ARTX_exog_tuned = ARXT_tree("exog", True, data)
    ARTX_exog_trained = ARXT_tree("exog", False, data)
    ARTX_target_tuned = ARXT_tree("target", True, data)
    ARTX_target_trained = ARXT_tree("target", False, data)
    ART_tuned = ART_tree(True, data)
    ART_trained = ART_tree(False, data)
    ARX_p_trained =  ARX_model(5, True, data)
    ARX_p =  ARX_model(5, False, data)
    AR_p =  AR_model(5, data)
    
    # plt.plot(data.iloc[1000:,0], label="truth")
    # plt.plot(ARTX_exog_tuned, label="ARX_p_trained")
    # plt.legend()
    # plt.show()

    if differencing: prep_met = "diff"
    else: prep_met = "norm"
    pd.DataFrame([ARTX_exog_tuned, ARTX_exog_trained, ARTX_target_tuned, ARTX_target_trained, ART_tuned, ART_trained, ARX_p_trained, ARX_p, AR_p]).to_csv(f"Data\\results_{prep_met}.csv")


if __name__ == "__main__":
    main()