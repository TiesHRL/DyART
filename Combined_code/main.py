import ARXT
import Data_gen
import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
import logging

def get_data():

    tickers = ["^GSPC", "^IXIC", "^DJI","JPYUSD=X", "^VIX"] # , "GBPUSD=X", "EURUSD=X",
    data = Data_gen.collect_data(tickers)
    return(data)

def train_run_tree(data, max_p, max_depth, min_size):

    d_val_cumsum, valid_prediction_list_cumsum, tree_list, hit_rate_ART_list, rmse_ART_list = ARXT.time_series_pred(data=data, max_p=max_p, preprocessing_method='normalization', max_depth=max_depth, min_size=min_size)
    average_rmse = np.mean(rmse_ART_list)
    average_hit_rate = np.mean(hit_rate_ART_list)

    return d_val_cumsum, valid_prediction_list_cumsum, tree_list, average_rmse, average_hit_rate
DATA = get_data()

def objective_function(max_p, max_depth, min_size):
    # Set up and train the ART model using the hyperparameters
    max_p, max_depth, min_size =  round(max_p), round(max_depth), round(min_size)
    _, _, _, average_rmse, average_hit_rate = train_run_tree(DATA[0:1000], max_p, max_depth, min_size)
    performance = average_hit_rate * 2 - average_rmse * 0.5
    return performance

def main():
    # Define hyperparameter bounds
    pbounds = {
        "max_p": (3, 20),
        "max_depth":(5, 30),
        "min_size":(3, 40)
    }

    # Create BayesianOptimization object
    optimizer = BayesianOptimization(f=objective_function, pbounds=pbounds, random_state=1)
    acq_function = UtilityFunction(kind="ei", kappa=5)
    # Run optimization
    optimizer.maximize(init_points=5, n_iter=10, acquisition_function = acq_function)
    
    # Print best results
    print(optimizer.max)

    return optimizer.max

logging.basicConfig(level=logging.INFO, filename='output.log', filemode='w')
logger = logging.getLogger(__name__)

logger.info("This is an info message")


if __name__ == "__main__":
    main()