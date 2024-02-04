import numpy as np
from scipy.stats import rankdata


def plugin_ece_est(confidences, labels, num_bins, p=2, debias=True):
    '''
    Input:
      confidences: (C_1, ... , C_n) \in [0, 1]^n
      labels: (Y_1, ..., Y_n) \in [0, 1]^n
      num_bins: m
      debias: If True, debias the plug-in estimator (only for p = 2)
    Output:
      Plug-in estimator of l_p-ECE(f)^p w.r.t. m equal-width bins
    '''
    # reindex to [0, min(num_bins, len(scores))]
    indexes = np.floor(num_bins * confidences).astype(int) 
    indexes = rankdata(indexes, method='dense') - 1
    counts = np.bincount(indexes)

    if p == 2 and debias:
        counts[counts < 2] = 2
        error = ((np.bincount(indexes, weights=confidences-labels)**2
              - np.bincount(indexes, weights=(confidences-labels)**2)) / (counts-1)).sum()
    else:
        counts[counts == 0] = 1
        error = (np.abs(np.bincount(indexes, weights=confidences-labels))**p / counts**(p - 1)).sum()

    return error / len(confidences)


def adaptive_ece_est(confidences, labels):
    '''
    Input:
      confidences: (C_1, ... , C_n) \in [0, 1]^n
      labels: (Y_1, ..., Y_n) \in [0, 1]^n
    Output:
      Adaptive debiased estimator of l_p-ECE(f)^2 using the dyadic grid of binning numbers
    '''

    num_bins_list = [2**b for b in range(1, np.floor(np.log2(len(confidences))-2).astype(int))]
    return np.max([plugin_ece_est(confidences, labels, num_bins, p=2, debias=True) for num_bins in num_bins_list])

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