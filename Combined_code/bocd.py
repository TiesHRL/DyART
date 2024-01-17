"""============================================================================
Author: Gregory Gundersen

Python implementation of Bayesian online changepoint detection for a normal
model with unknown mean parameter. For algorithm details, see

    Adams & MacKay 2007
    "Bayesian Online Changepoint Detection"
    https://arxiv.org/abs/0710.3742

For Bayesian inference details about the Gaussian, see:

    Murphy 2007
    "Conjugate Bayesian analysis of the Gaussian distribution"
    https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf

This code is associated with the following blog posts:

    http://gregorygundersen.com/blog/2019/08/13/bocd/
    http://gregorygundersen.com/blog/2020/10/20/implementing-bocd/
============================================================================"""

import matplotlib.pyplot as plt
from   matplotlib.colors import LogNorm
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from   scipy.stats import norm
from   scipy.special import logsumexp

# -----------------------------------------------------------------------------

def bocd_offline(data, model, hazard):
    """Return run length posterior using Algorithm 1 in Adams & MacKay 2007.
    """
    # 1. Initialize lower triangular matrix representing the posterior as
    #    function of time. Model parameters are initialized in the model class.
    #    
    #    When we exponentiate R at the end, exp(-inf) --> 0, which is nice for
    #    visualization.
    #
    T           = len(data)
    log_R       = -np.inf * np.ones((T+1, T+1))
    log_R[0, 0] = 0              # log 0 == 1
    pmean       = np.empty(T)    # Model's predictive mean.
    pvar        = np.empty(T)    # Model's predictive variance. 
    log_message = np.array([0])  # log 0 == 1
    log_H       = np.log(hazard)
    log_1mH     = np.log(1 - hazard)

    for t in range(1, T+1):
        # 2. Observe new datum.
        x = data[t-1]

        # Make model predictions.
        pmean[t-1] = np.sum(np.exp(log_R[t-1, :t]) * model.mean_params[:t])
        pvar[t-1]  = np.sum(np.exp(log_R[t-1, :t]) * model.var_params[:t])
        
        # 3. Evaluate predictive probabilities.
        log_pis = model.log_pred_prob(t, x)

        # 4. Calculate growth probabilities.
        log_growth_probs = log_pis + log_message + log_1mH

        # 5. Calculate changepoint probabilities.
        log_cp_prob = logsumexp(log_pis + log_message + log_H)

        # 6. Calculate evidence
        new_log_joint = np.append(log_cp_prob, log_growth_probs)

        # 7. Determine run length distribution.
        log_R[t, :t+1]  = new_log_joint
        log_R[t, :t+1] -= logsumexp(new_log_joint)

        # 8. Update sufficient statistics.
        model.update_params(t, x)

        # Pass message.
        log_message = new_log_joint

    R = np.exp(log_R)
    return R, pmean, pvar

# -----------------------------------------------------------------------------
class BOCD_Online:
    def __init__(self, T, model, hazard):
        self.T           = T
        self.log_R       = -np.inf * np.ones((self.T+1, self.T+1))
        self.log_R[0, 0] = 0              # log 0 == 1
        self.pmean       = np.empty(self.T)    # Model's predictive mean.
        self.pvar        = np.empty(self.T)    # Model's predictive variance. 
        self.log_message = np.array([0])  # log 0 == 1
        self.log_H       = np.log(hazard)
        self.log_1mH     = np.log(1 - hazard)
        self.model       = model

        self.R = np.exp(self.log_R)
        self.changepoints = []
    def update(self, i, x):

        # 2. Observe new datum.
        # Make model predictions.
        self.pmean[i-1] = np.sum(np.exp(self.log_R[i-1, :i]) * self.model.mean_params[:i])
        self.pvar[i-1]  = np.sum(np.exp(self.log_R[i-1, :i]) * self.model.var_params[:i])
        
        # 3. Evaluate predictive probabilities.
        log_pis = self.model.log_pred_prob(i, x)

        # 4. Calculate growth probabilities.
        log_growth_probs = log_pis + self.log_message + self.log_1mH

        # 5. Calculate changepoint probabilities.
        log_cp_prob = logsumexp(log_pis + self.log_message + self.log_H)

        # 6. Calculate evidence
        new_log_joint = np.append(log_cp_prob, log_growth_probs)

        # 7. Determine run length distribution.
        self.log_R[i, :i+1]  = new_log_joint
        self.log_R[i, :i+1] -= logsumexp(new_log_joint)

        # 8. Update sufficient statistics.
        self.model.update_params(i, x)

        # Pass message.
        self.log_message = new_log_joint

        if np.exp(self.log_R[i, 1]) > 0.8:
            try:
                if i - self.changepoints[-1] >= 250:
                    self.changepoints.append(i)
                    return True
            except:
                if i >= 100:
                    self.changepoints.append(i)
                    return True            
        else:
            return False        
    def retR(self):
        return np.exp(self.log_R)
    
# -----------------------------------------------------------------------------


class GaussianUnknownMean:
    
    def __init__(self, mean0, var0, varx):
        """Initialize model.
        
        meanx is unknown; varx is known
        p(meanx) = N(mean0, var0)
        p(x) = N(meanx, varx)
        """
        self.mean0 = mean0
        self.var0  = var0
        self.varx  = varx
        self.mean_params = np.array([mean0])
        self.prec_params = np.array([1/var0])
    
    def log_pred_prob(self, t, x):
        """Compute predictive probabilities \pi, i.e. the posterior predictive
        for each run length hypothesis.
        """
        # Posterior predictive: see eq. 40 in (Murphy 2007).
        post_means = self.mean_params[:t]
        post_stds  = np.sqrt(self.var_params[:t])
        return norm(post_means, post_stds).logpdf(x)
    
    def update_params(self, t, x):
        """Upon observing a new datum x at time t, update all run length 
        hypotheses.
        """
        # See eq. 19 in (Murphy 2007).
        new_prec_params  = self.prec_params + (1/self.varx)
        self.prec_params = np.append([1/self.var0], new_prec_params)
        # See eq. 24 in (Murphy 2007).
        new_mean_params  = (self.mean_params * self.prec_params[:-1] + \
                            (x / self.varx)) / new_prec_params
        self.mean_params = np.append([self.mean0], new_mean_params)

    @property
    def var_params(self):
        """Helper function for computing the posterior variance.
        """
        return 1./self.prec_params + self.varx

# -----------------------------------------------------------------------------

def generate_data(varx, mean0, var0, T, cp_prob):
    """Generate partitioned data of T observations according to constant
    changepoint probability `cp_prob` with hyperpriors `mean0` and `prec0`.
    """
    data  = []
    cps   = []
    meanx = mean0
    for t in range(0, T):
        if np.random.random() < cp_prob:
            meanx = np.random.normal(mean0, var0)
            cps.append(t)
        data.append(np.random.normal(meanx, varx))
    return data, cps


# -----------------------------------------------------------------------------

def plot_posterior(T, data, R, pmean, pvar, cps, dts):
    # Assuming R is already truncated as needed and converted to a DataFrame
    R = R[:-1,:-1]
    dts = pd.to_datetime(dts)

    R = pd.DataFrame(R, index=dts)
    
    fig, axes = plt.subplots(2, 1, figsize=(20, 10), sharex=True)
    ax1, ax2 = axes

    ax1.scatter(dts, data)
    ax1.plot(dts, data)

    # Plot predictions.
    ax1.plot(dts, pmean, color='k')
    _2std = 2 * np.sqrt(pvar)
    ax1.plot(dts, pmean - _2std, color='k', linestyle='--')
    ax1.plot(dts, pmean + _2std, color='k', linestyle='--')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Set x-axis ticks per year
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))


    dts_num = mdates.date2num(dts)

    # Now use dts_num for the extent parameter
    ax2.imshow(np.rot90(R), aspect='auto', cmap='gray_r', 
            norm=LogNorm(vmin=0.0001, vmax=1), 
            extent=[dts_num[0], dts_num[-1], 1, R.shape[0]])
    # Rotate the matrix R by 90 degrees without flipping vertically.
    # ax2.imshow((np.rot90(R)), aspect='auto', cmap='gray_r', 
    #            norm=LogNorm(vmin=0.0001, vmax=1), extent=[0, T, 1, 4000])

    # Set y-axis limits for ax2
    ax2.set_ylim([1, 4000])

    ax2.margins(0)
    changepoints = data.index[cps]
    changepoints = mdates.date2num(changepoints)


    for cp in changepoints:
        ax1.axvline(cp, c='red', ls='dotted')
        ax2.axvline(cp, c='red', ls='dotted')

    plt.tight_layout()
    plt.show()

from scipy.signal import argrelextrema

def get_changepoint_indices(R, threshold=0.8, min_distance=100):
    # R is the run length probability matrix from the BOCD algorithm
    # threshold is the minimum probability to consider for a changepoint
    changepoint_indices = np.where(R[:, 1] > threshold)[0]
    changepoint_indices.sort()

    # Filter out changepoints that are too close to each other
    filtered_cps = []
    last_cp = -min_distance  # Initialize with a value min_distance less than the first index
    
    for cp in changepoint_indices:
        if cp - last_cp >= min_distance:
            filtered_cps.append(cp)
            last_cp = cp
    if 1 in filtered_cps: filtered_cps = filtered_cps[1:]

    return filtered_cps

# Using the above function with the R matrix from your code
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    train_len = 1000

    data = pd.read_csv('Data/fin_data.csv')
    data.index = data["Date"]
    data = data.drop(["Date"], axis=1)
    dt_index = data.index[train_len:]
    data = data.pct_change()
    data = data.iloc[:,0]    
    dt_index = data.index[train_len:]
    print(data)
    # data = np.log1p(data.pct_change())
    # print(data)

    data = data*100
    # data = (data - np.mean(data)) / np.std(data)
    T      = len(data)-train_len   # Number of observations.
    hazard = 1/500  # Constant prior on changepoint probability.
    mean0  = np.mean(data[0:train_len])      # The prior mean on the mean parameter.
    var0   = 1  # The prior variance for mean parameter.
    varx   = np.std(data[0:train_len])   # The known variance of the data.
    data = data[train_len:]
    print(mean0, var0, varx)


    model          = GaussianUnknownMean(mean0, var0, varx)
    R, pmean, pvar = bocd_offline(data, model, hazard)
   
    cps = get_changepoint_indices(R, min_distance=250)
    changepoints = data.index[cps]
    print(changepoints)
    pd.DataFrame(changepoints).to_csv("Data/changepoints.csv", index=False)
    # pd.DataFrame(R).to_csv("R_mat.csv", index=False)
    
    plot_posterior(T, data, R, pmean, pvar, cps, dt_index)