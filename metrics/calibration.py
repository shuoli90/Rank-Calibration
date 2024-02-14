import numpy as np
from scipy.stats import rankdata
import warnings


def plugin_ece_est(confidences, correctness, num_bins, p=2, debias=True):
    '''
    Input:
        confidences: (C_1, ... , C_n) \in [0, 1]^n
        correctness: (A_1, ..., A_n) \in [0, 1]^n
        num_bins: B
        debias: If True, debias the plug-in estimator (only for p = 2)
    Output:
        Plug-in estimator of l_p-ECE(f)^p w.r.t. B equal-width bins
    '''
    # reindex to [0, min(num_bins, len(scores))]
    indexes = np.floor(num_bins * confidences).astype(int) 
    indexes = rankdata(indexes, method='dense') - 1
    counts = np.bincount(indexes)

    if p == 2 and debias:
        counts[counts < 2] = 2
        error = ((np.bincount(indexes, weights=confidences-correctness)**2
              - np.bincount(indexes, weights=(confidences-correctness)**2)) / (counts-1)).sum()
    else:
        counts[counts == 0] = 1
        error = (np.abs(np.bincount(indexes, weights=confidences-correctness))**p / counts**(p - 1)).sum()

    return error / len(confidences)


def adaptive_ece_est(confidences, correctness):
    '''
    Input:
        confidences: (C_1, ... , C_n) \in [0, 1]^n
        correctness: (A_1, ..., A_n) \in [0, 1]^n
    Output:
        Adaptive debiased estimator of l_p-ECE(f)^2 using the dyadic grid of binning numbers
    '''

    num_bins_list = [2**b for b in range(1, np.floor(np.log2(len(confidences))-2).astype(int))]
    return np.max([plugin_ece_est(confidences, correctness, num_bins, p=2, debias=True) for num_bins in num_bins_list])

class ECE_estimate():

    def __init__(self, metric_name='ADPE'):
        self.metric_name = metric_name
        if metric_name not in ['PE', 'PE2', 'DPE', 'ADPE']:
            raise ValueError("Please specify a valid calibration metric!")
        self.metric = {'PE': lambda c, y, B: plugin_ece_est(c, y, B, 1, False),\
                        'PE2': lambda c, y, B: plugin_ece_est(c, y, B, 2, False),\
                        'DPE': lambda c, y, B: plugin_ece_est(c, y, B, 2, True),\
                        'ADPE': lambda c, y: adaptive_ece_est(c, y)}[metric_name]

    def __call__(self, confidences, labels, num_bins=None):
        if self.metric_name == 'ADPE':
            return self.metric(confidences, labels)
        else:
            return self.metric(confidences, labels, num_bins)

def AUARC(confidences, labels):
    # An accuracy rejection curve (ARC) is a function representating the accuracy of a classifier \
    # as a function of its reject rate. 
    # An ARC is therefore produced by plotting the accuracy of a classifier against its reject rate, from [0,1].
    # All ARCs have an accuracy of 100% for a rejection rate of 100%
    rejection_rate = np.arange(0, 1, 0.01)
    if type(confidences) is not np.ndarray:
        confidences = np.array(confidences)
    if type(labels) is not np.ndarray:
        labels = np.array(labels)
    arcs = []
    for rate in rejection_rate:
        labels_tmp = labels[confidences > np.quantile(confidences, rate)]
        if len(labels_tmp) == 0:
            arc = 1.0
        else:
            # reject rate portion of the data
            arc = np.mean(labels_tmp)
        arcs.append(arc)
    return np.trapz(arcs, rejection_rate)

def plugin_erce_est(uncertainties, correctness, num_bins=20, p=1):
    '''
    Input:
        uncertainties: (U_1, ... , U_n) \in R^n
        correctness: (A_1, ..., A_n) \in [0, 1]^n
        num_bins: B
    Output:
        Plug-in estimator of l_p-ERCE(f)^p w.r.t. B equal-mass bins
    '''
    n = len(correctness)
    bin_endpoints = [round(ele) for ele in np.linspace(0, n, num_bins+1)]
    sorted_indices = np.argsort(uncertainties)
    correctness = correctness[sorted_indices]
    uncertainties = uncertainties[sorted_indices]
    a_map = -np.ones_like(correctness)
    u_map = np.zeros_like(uncertainties)
    a_hats = []
    u_hats = []
    # compute a_hat, u_hat and a_map: i -> a_hat, u_map: i -> u_hat
    for idx_bin in range(1, num_bins+1):
        lo, hi = bin_endpoints[idx_bin-1], bin_endpoints[idx_bin]
        if hi > lo:
            # if idx_bin == num_bins:  hi += 1
            bin_correctness = correctness[lo:hi]
            a_hat = np.mean(bin_correctness)
            a_map[lo:hi] = a_hat
            u_hat = np.mean(uncertainties[lo:hi]) if hi > lo else None
            u_map[lo:hi] = u_hat
            a_hats.append(a_hat)
            u_hats.append(u_hat)
    # breakpoint()
    result = 0
    for a_hat, u_hat in zip(a_hats, u_hats):

        count_correnct = (np.sum(a_map >= a_hat) - (1+np.sum(a_map == a_hat)) / 2) / (n-1)
        count_uncertainty = (np.sum(u_map <= u_hat) - (1+np.sum(u_map == u_hat)) / 2) / (n-1)
        if p == 1:
            tmp = np.abs(count_correnct - count_uncertainty)
        elif p == 2:
            tmp = (count_correnct - count_uncertainty) ** 2
        else:
            raise ValueError("Please specify a valid order p!")
        result += tmp * np.sum(u_map == u_hat) / n
        # breakpoint()
    return result

    