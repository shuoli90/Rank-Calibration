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
    a_map = -np.ones(n)
    u_map = np.zeros(n)
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
    result = 0
    for a_hat, u_hat in zip(a_hats, u_hats):

        count_correnct = (np.sum(a_map >= a_hat) - (1+np.sum(a_map == a_hat)) / 2) / (n-1)
        count_uncertainty = (np.sum(u_map <= u_hat) - (1+np.sum(u_map == u_hat)) / 2) / (n-1)
        if p == 1:
            result += np.abs(count_correnct - count_uncertainty) * np.sum(u_map == u_hat) / n
        elif p == 2:
            result += (count_correnct - count_uncertainty) ** 2 * np.sum(u_map == u_hat) / n
        else:
            raise ValueError("Please specify a valid order p!")
    return result

def rank_erce_est(uncertainties, correctness, num_bins=20, p=1):
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
    a_rank_map = np.zeros(num_bins)
    u_rank_map = np.zeros(num_bins)
    num_count = np.zeros(num_bins)
    # compute cdf of correctness
    correct_ranks, uncertainty_ranks = np.zeros(n), np.zeros(n)
    for i in range(n):
        correct_ranks[i] = (np.sum(correctness[i] >= correctness)-1) / (n-1)
        uncertainty_ranks[i] = (np.sum(uncertainties[i] >= uncertainties)-1) / (n-1)
    # compute a_hat, u_hat and a_map: i -> a_hat, u_map: i -> u_hat
    for idx_bin in range(1, num_bins+1):
        lo, hi = bin_endpoints[idx_bin-1], bin_endpoints[idx_bin]
        if hi > lo:
            bin_correct_ranks = correct_ranks[lo:hi]
            a_rank_hat = np.mean(bin_correct_ranks)
            a_rank_map[idx_bin-1] = a_rank_hat
            u_rank_hat = np.mean(uncertainty_ranks[lo:hi])
            u_rank_map[idx_bin-1] = u_rank_hat
        num_count[idx_bin-1] = hi-lo

    result = 0
    for idx in range(num_bins):
        tmp = a_rank_map[idx] - (1-u_rank_map[idx])
        if p == 1:
            result += np.abs(tmp) * num_count[idx] / n
        elif p == 2:     
            result += tmp ** 2 * num_count[idx] / n
        else:
            raise ValueError("Please specify a valid order p!")
    return result

def debias_rank_erce_est(uncertainties, correctness, num_bins=20):
    '''
    Input:
        uncertainties: (U_1, ... , U_n) \in R^n
        correctness: (A_1, ..., A_n) \in [0, 1]^n
        num_bins: B
    Output:
        Plug-in estimator of l_2-Rank-ERCE(f)^2 w.r.t. B equal-mass bins
    '''
    n = len(correctness)
    bin_endpoints = [round(ele) for ele in np.linspace(0, n, num_bins+1)]
    sorted_indices = np.argsort(uncertainties)
    correctness = correctness[sorted_indices]
    uncertainties = uncertainties[sorted_indices]
    # compute cdf of correctness
    correct_ranks, uncertainty_ranks = np.zeros(n), np.zeros(n)
    for i in range(n):
        correct_ranks[i] = (np.sum(correctness[i] >= correctness)-1)/(n-1)
        # breakpoint()
        uncertainty_ranks[i] = (np.sum(uncertainties[i] >= uncertainties)-1)/(n-1)
    # breakpoint()
    result = 0
    for idx_bin in range(1, num_bins+1):
        lo, hi = bin_endpoints[idx_bin-1], bin_endpoints[idx_bin]
        for i in range(lo, hi):
            for j in range(i+1, hi):
                correct_rank_i, correct_rank_j \
                    = correct_ranks[i]+(correct_ranks[i]-(correctness[i]<=correctness[j]))/(n-2),\
                    correct_ranks[j]+(correct_ranks[j]-(correctness[j]<=correctness[i]))/(n-2)
                uncertainty_rank_i, uncertainty_rank_j \
                    = uncertainty_ranks[i]+(uncertainty_ranks[i]-(uncertainties[i]<=uncertainties[j]))/(n-2),\
                    uncertainty_ranks[j]+(uncertainty_ranks[j]-(uncertainties[j]<=uncertainties[i]))/(n-2)
                result += 2*(correct_rank_i-1+uncertainty_rank_i)*(correct_rank_j-1+uncertainty_rank_j)/((n-1)*(hi-lo))
    return result
    
def adaptive_rank_erce_est(uncertainties, correctness):
    '''
    Input:
        uncertainties: (U_1, ... , U_n) \in R^n
        correctness: (A_1, ..., A_n) \in [0, 1]^n
        num_bins: B
    Output:
        Plug-in estimator of l_2-Rank-ERCE(f)^2 w.r.t. B equal-mass bins
    '''
    n = len(correctness)
    num_bins_list = [2**b for b in range(2, np.floor(np.log2(n)-2).astype(int))]
    if not num_bins_list:
        raise ValueError("The evaluation dataset is too small!")
    return np.max([debias_rank_erce_est(uncertainties, correctness, num_bins) for num_bins in num_bins_list])

def nested_rank_erce_est(uncertainties, correctness, num_bins=20, p=1):
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
    a_map = np.zeros(num_bins)
    u_map = np.zeros(num_bins)
    num_count = np.zeros(num_bins)
    # compute cdf of correctness
    correct_ranks, uncertainty_ranks = np.zeros(n), np.zeros(n)
    for i in range(n):
        correct_ranks[i] = (np.sum(correctness[i] >= correctness)-1) / (n-1)
        uncertainty_ranks[i] = (np.sum(uncertainties[i] >= uncertainties)-1) / (n-1)
    for idx_bin in range(1, num_bins+1):
        lo, hi = bin_endpoints[idx_bin-1], bin_endpoints[idx_bin]
        if hi > lo:
            bin_correct_ranks = correct_ranks[lo:hi]
            a_hat = np.mean(bin_correct_ranks)
            a_map[idx_bin-1] = a_hat
            u_hat = np.mean(uncertainty_ranks[lo:hi])
            u_map[idx_bin-1] = u_hat
        num_count[idx_bin-1] = hi-lo

    a_map_ranks = np.zeros(num_bins)
    u_map_ranks = np.zeros(num_bins)
    for idx in range(num_bins):
        a_map_ranks[idx] = (np.sum(a_map[idx] >= a_map)-1) / (num_bins-1)
        u_map_ranks[idx] = (np.sum(u_map[idx] >= u_map)-1) / (num_bins-1)

    result = 0
    for idx in range(num_bins):
        tmp = a_map_ranks[idx] - (1-u_map_ranks[idx])
        if p == 1:
            result += np.abs(tmp) * num_count[idx] / n
        elif p == 2:     
            result += tmp ** 2 * num_count[idx] / n
        else:
            raise ValueError("Please specify a valid order p!")
    return result