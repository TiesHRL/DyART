from networkx import difference
import ARXT
from ARXT import hit_rate
import data_gen
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
from bayes_opt import BayesianOptimization, UtilityFunction
from statsmodels.tsa.arima.model import ARIMA
import time 
from bocd import BOCD_Online, GaussianUnknownMean
def get_data(differencing = False):
    """
    Retrieves and preprocesses financial data.

    :param differencing: Boolean to determine if differencing is applied.
    :return: Preprocessed data as a Pandas DataFrame.
    """
    # tickers = ["^GSPC", "^IXIC", "^DJI","JPYUSD=X", "^VIX", "GBPUSD=X", "EURUSD=X]
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
    """
    Trains and evaluates the ARXT model.

    :param data: The input data.
    :param p: The lag order.
    :param max_depth: The maximum depth of the tree.
    :param min_size: The minimum size of a node.
    :param max_weight: The maximum weight.
    :param splt: Split method.
    :return: Cumulative actual values, predictions, list of trees, hit rate, and RMSE.
    """
    p, max_depth, min_size =  int(round(p)), int(round(max_depth)), int(round(min_size))
    d_val_cumsum, valid_prediction_cumsum, tree_list = ARXT.ARXT_time_series_pred(data=data, p=p, preprocessing_method='normalization', max_depth=max_depth, min_size=min_size, max_weight=max_weight, splt=splt)
    hit_rate_sample = hit_rate(d_val_cumsum, valid_prediction_cumsum)

    rmse_sample = sqrt(mean_squared_error(d_val_cumsum, valid_prediction_cumsum))

    return d_val_cumsum, valid_prediction_cumsum, tree_list, hit_rate_sample, rmse_sample

def train_run_ART(data, p, max_depth, min_size):
    """
    Trains and evaluates the ART model.

    :param data: The input data.
    :param p: The lag order.
    :param max_depth: The maximum depth of the tree.
    :param min_size: The minimum size of a node.
    :return: Cumulative actual values, predictions, list of trees, hit rate, and RMSE.
    """
    p, max_depth, min_size =  int(round(p)), int(round(max_depth)), int(round(min_size))
    d_val_cumsum, valid_prediction_cumsum, tree_list = ARXT.ART_time_series_pred(data=data, p=p, preprocessing_method='normalization', max_depth=max_depth, min_size=min_size)
    hit_rate_sample = hit_rate(d_val_cumsum, valid_prediction_cumsum)

    rmse_sample = sqrt(mean_squared_error(d_val_cumsum, valid_prediction_cumsum))

    return d_val_cumsum, valid_prediction_cumsum, tree_list, hit_rate_sample, rmse_sample

# define and load in to global variables 
CPS = pd.read_csv('data/changepoints.csv').to_numpy()

def objective_function(p, max_depth, min_size, max_weight, start, fin, splt, data, ART_bool):
    """
    Defines the objective function to pass to the optimizer function

    :param data: The input data.
    :param p: The lag order.
    :param max_depth: The maximum depth of the tree.
    :param min_size: The minimum size of a node.
    :param start: Starting point to be used by the optimisation.
    :param fin: Finishing point to be used by the optimisation.
    :param splt: If splits should be made on exogenous variables.
    :param data: Dataframe containing data with which the optimisation takes place.
    :param ART_bool: Boolean to tell the model if an ART or ARXT model is to be trained.
    :return: Performence Metric.
    """
    if ART_bool:
        p, max_depth, min_size =  round(p), round(max_depth), round(min_size)
        _, _, _, hit_rate_sample, rmse_sample = train_run_ART(data[start:fin], p, max_depth, min_size)

    else:
        p, max_depth, min_size, max_weight =  round(p), round(max_depth), round(min_size), max_weight
        _, _, _, hit_rate_sample, rmse_sample = train_run_tree(data[start:fin], p, max_depth, min_size, max_weight, splt=splt)

    performance = 2 * hit_rate_sample - rmse_sample * 0.5
    return performance

def optimizer(pbounds, start, fin, splt, data, ART_bool, init_points=10, n_iter=30):
    """
    Conducts optimization using Bayesian Optimization to find the best hyperparameters.

    :param pbounds: Dictionary containing the bounds for the parameters to be optimized.
    :param start: Starting index for the data to be used in optimization.
    :param fin: Ending index for the data to be used in optimization.
    :param splt: Specifies if splits should be made on exogenous variables.
    :param data: Dataframe containing the data for optimization.
    :param ART_bool: Boolean indicating if the model is ART (True) or ARXT (False).
    :param init_points: Number of initial random points for the Bayesian optimizer.
    :param n_iter: Number of iterations for the optimization process.
    :return: Dictionary containing the optimized parameters.
    """
    if ART_bool:
        optimizer = BayesianOptimization(f= lambda p, max_depth, min_size: objective_function(p, max_depth, min_size, 0, start, fin, splt, data, ART_bool), pbounds=pbounds, random_state=1)
    else:
        optimizer = BayesianOptimization(f= lambda p, max_depth, min_size, max_weight: objective_function(p, max_depth, min_size, max_weight, start, fin, splt, data, ART_bool), pbounds=pbounds, random_state=1)
    acq_function = UtilityFunction(kind="ei", kappa=5, kappa_decay=0.8)
    optimizer.maximize(init_points, n_iter, acquisition_function = acq_function)
    opt_params  = optimizer.max['params']
    return opt_params

def ART_tree(tune, train, data):
    """
    Constructs and forecasts using an Autoregressive Tree (ART) model.

    :param tune: Boolean indicating if the model should be retuned.
    :param train: Boolean indicating if the model should be retrained.
    :param data: Dataframe containing the dataset for model training and forecasting.
    :return: List of forecasts generated by the ART model.
    """    
    ART_bool = True
    splt = "none"
    start_time = time.time()
    train_len = 1000
    # Define hyperparameter bounds
    if tune: retrain = "retune"
    elif train: retrain = "retrain"
    else: retrain = ""
    pbounds = {
        "p": (1, 20),
        "max_depth":(10, 150),
        "min_size":(1, 50)
        }

    opt_params = optimizer(pbounds, 0 , train_len, splt, data, ART_bool)
    p, max_depth, min_size = round(opt_params['p']), round(opt_params['max_depth']), round(opt_params['min_size'])
    next_pbounds = {"p": (p*0.7, p*1.3), "max_depth" : (max_depth*0.7, max_depth*1.3), "min_size" : (min_size*0.7, min_size*1.3)}

    ART = ARXT.AutoregressiveTree(p)    

    _, _, tree, _, _ = train_run_ART(data=data.iloc[:train_len], p=p, max_depth=max_depth, min_size=min_size)

    T      = len(data.iloc[:,0])-train_len   # Number of observations.
    hazard = 1/500  # Constant prior on changepoint probability.
    mean0  = np.mean(data.iloc[0:train_len,0])      # The prior mean on the mean parameter.
    var0   = 1  # The prior variance for mean parameter.
    varx   = np.std(data.iloc[0:train_len,0])  # The known variance of the data.
    model = GaussianUnknownMean(mean0, var0, varx)
    CPD = BOCD_Online(T, model, hazard)

    forecasts = []
    for i in range(train_len, len(data[0])):
        forecasts.append(ARXT.forecast_ART(data.iloc[i-1000:i], tree, ART, p))
        if CPD.update(i-train_len+1, data.iloc[i, 0]):
            if tune:
                print("retuning at ", data.index[i])
                opt_params = optimizer(next_pbounds, i-500, i, splt, data, ART_bool, init_points=5, n_iter = 10)
                p, max_depth, min_size, = round(opt_params['p']), round(opt_params['max_depth']), round(opt_params['min_size'])
                next_pbounds = {"p": (p*0.7, p*1.3), "max_depth" : (max_depth*0.7, max_depth*1.3), "min_size" : (min_size*0.7, min_size*1.3)}
            if train:
                ART = ARXT.AutoregressiveTree(p)    
                _, _, tree, _, _ = train_run_ART(data=data[i-600:i], p=p, max_depth=max_depth, min_size=min_size)

    end_time = time.time()
    duration = end_time-start_time
    print("Time taken for ART {} : {} mins".format(retrain, round(duration)/60))
    return forecasts

def ARXT_tree(splt, tune, train, data):
    """
    Constructs and forecasts using an Autoregressive eXogenous Tree (ARXT) model.

    :param splt: Specifies the splitting strategy for exogenous variables.
    :param tune: Boolean indicating if the model should be retuned.
    :param train: Boolean indicating if the model should be retrained.
    :param data: Dataframe containing the dataset for model training and forecasting.
    :return: List of forecasts generated by the ARXT model.
    """
    start_time = time.time()
    train_len = 1000
    ART_bool = False

    if tune: retrain = "retune"
    elif train: retrain = "retrain"
    else: retrain = ""

    pbounds = {
        "p": (1, 20),
        "max_depth":(10, 150),
        "min_size":(1, 50),
        "max_weight": (0.01, 0.15)
    }

    opt_params = optimizer(pbounds, 0 , train_len, splt, data, ART_bool)
    p, max_depth, min_size, max_weight = round(opt_params['p']), round(opt_params['max_depth']), round(opt_params['min_size']), opt_params['max_weight']
    next_pbounds = {"p": (p*0.7, p*1.3), "max_depth" : (max_depth*0.7, max_depth*1.3), "min_size" : (min_size*0.7, min_size*1.3), "max_weight" : (max(0.001, max_weight*0.7), max_weight*1.3)}

    ART = ARXT.AutoregressiveXTree(p, splt=splt)    

    _, _, tree, _, _ = train_run_tree(data=data.iloc[:train_len], p=p, max_depth=max_depth, min_size=min_size, max_weight=max_weight, splt=splt)
    
    T      = len(data.iloc[:,0])-train_len   # Number of observations.
    hazard = 1/500  # Constant prior on changepoint probability.
    mean0  = np.mean(data.iloc[0:train_len,0])      # The prior mean on the mean parameter.
    var0   = 1  # The prior variance for mean parameter.
    varx   = np.std(data.iloc[0:train_len,0]) # The known variance of the data.
    model = GaussianUnknownMean(mean0, var0, varx)
    CPD = BOCD_Online(T, model, hazard)

    forecasts = []
    for i in range(train_len, len(data[0])):
        forecasts.append(ARXT.forecast(data.iloc[i-200:i], tree, ART, p))
        if CPD.update(i-train_len+1, data.iloc[i, 0]):
            if tune:
                print("retraining at ", data.index[i])
                opt_params = optimizer(next_pbounds, i-500, i, splt, data, ART_bool, init_points=5, n_iter = 10)
                p, max_depth, min_size, max_weight = round(opt_params['p']), round(opt_params['max_depth']), round(opt_params['min_size']), opt_params['max_weight']
                next_pbounds = {"p": (p*0.7, p*1.3), "max_depth" : (max_depth*0.7, max_depth*1.3), "min_size" : (min_size*0.7, min_size*1.3), "max_weight" : (max(0.001, max_weight*0.7), max_weight*1.3)}
            if train:
                ART = ARXT.AutoregressiveXTree(p, splt=splt)    
                _, _, tree, _, _ = train_run_tree(data=data[i-600:i], p=p, max_depth=max_depth, min_size=min_size, max_weight=max_weight, splt=splt)
    end_time = time.time()
    duration = end_time-start_time
    print("Time taken for ARXT {} {}: {} mins".format(splt, retrain, round(duration)/60))
    return forecasts

def ARX_model(p, train, data):
    """
    Constructs and forecasts using an Autoregressive with eXogenous inputs (ARX) model.

    :param p: The order of the autoregressive model.
    :param train: Boolean indicating if the model should be retrained.
    :param data: Dataframe containing the dataset for model training and forecasting.
    :return: List of forecasts generated by the ARX model.
    """
    start_time = time.time()
    train_len = 1000
    AR = ARXT.ARX_p(data, p)    

    AR.ARX_p_model(0, train_len)

    T      = len(data.iloc[:,0])-train_len   # Number of observations.
    hazard = 1/500  # Constant prior on changepoint probability.
    mean0  = np.mean(data.iloc[0:train_len,0])      # The prior mean on the mean parameter.
    var0   = 1  # The prior variance for mean parameter.
    varx   = np.std(data.iloc[0:train_len,0])  # The known variance of the data.
    model = GaussianUnknownMean(mean0, var0, varx)
    CPD = BOCD_Online(T, model, hazard)

    forecasts = []
    for i in range(train_len, len(data[0])):
        if train:
            if CPD.update(i-train_len+1, data.iloc[i, 0]):
                print("retraining at ", data.index[i])
                AR.ARX_p_model(max(0,i-500), i)

        forecasts.append(AR.predict(data[i-p:i]))
    retrain = ""
    if train: retrain = "retune"
    end_time = time.time()
    duration = end_time-start_time
    print("Time taken for ARX(p) {}: {} mins".format(retrain, round(duration)/60))

    return forecasts
def AR_model(p, data):
    """
    Constructs and forecasts using an Autoregressive (AR) model.

    :param p: The order of the autoregressive model.
    :param data: Dataframe containing the dataset for model training and forecasting.
    :return: List of forecasts generated by the AR model.
    """
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
    """
    The main function that executes the entire forecasting process.

    This function:
    - Retrieves the data,
    - Executes different forecasting models (ARXT with exogenous/target variables, ART, ARX, AR),
    - Optionally tunes and retrains the models,
    - Outputs the forecasts to a CSV file.

    The function does not take any parameters and does not return any values. It primarily serves to orchestrate the forecasting workflow.
    """
    differencing = False
    data = get_data(differencing=differencing)
    print(data)

    # Calling various forecasting functions with different configurations
    # ARXT_exog_tuned = ARXT_tree("exog", True, True, data)
    # ARXT_exog_trained = ARXT_tree("exog", False, True, data)
    # ARXT_exog = ARXT_tree("exog", False, False, data)
    # ARXT_target_tuned = ARXT_tree("target", True, True, data)
    # ARXT_target_trained = ARXT_tree("target", False, True, data)
    # ARXT_target = ARXT_tree("target", False, False, data)
    ART_tuned = ART_tree(True, True, data)
    # ART_trained = ART_tree(False, True, data)
    # ART = ART_tree(False, False, data)
    # ARX_p_trained =  ARX_model(5, True, data)
    # ARX_p =  ARX_model(5, False, data)
    # AR_p =  AR_model(5, data)
    
    if differencing: prep_met = "diff"
    else: prep_met = "norm"
    # pd.DataFrame([ARXT_exog_tuned, ARXT_exog_trained, ARXT_exog, ARXT_target_tuned, ARXT_target_trained, ARXT_target, ART_tuned, ART_trained, ART, ARX_p_trained, ARX_p, AR_p]).to_csv(f"Data\\results_{prep_met}.csv")

if __name__ == "__main__":
    main()