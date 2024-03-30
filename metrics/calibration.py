import numpy as np
from scipy.stats import rankdata
import warnings
from scipy.stats import norm
from sklearn.metrics import precision_recall_curve, auc
import bisect

def plugin_ece_est(correctness, confidences, num_bins, p=2, debias=True):
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


def adaptive_ece_est(correctness, confidences):
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
        self.metric = {'PE': lambda c, y, B: plugin_ece_est(y, c, B, 1, False),\
                        'PE2': lambda c, y, B: plugin_ece_est(y, c, B, 2, False),\
                        'DPE': lambda c, y, B: plugin_ece_est(y, c, B, 2, True),\
                        'ADPE': lambda c, y: adaptive_ece_est(y, c)}[metric_name]

    def __call__(self, labels, confidences, num_bins=None):
        if self.metric_name == 'ADPE':
            return self.metric(labels, confidences, )
        else:
            return self.metric(labels, confidences, num_bins)

def AUARC(labels, confidences):
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

def AUPRC(labels, confidences):
    # An precision recall curve (PRC) is a function representating the precision of a classifier \
    # as a function of its recall. 
    min_conf, max_conf = np.min(confidences), np.max(confidences)
    
    # Perform min-max normalization; do not affet AUPRC values
    normalized_confidences = (confidences - min_conf) / (max_conf - min_conf)
    precision, recall, _ = precision_recall_curve(labels, normalized_confidences)
    
    # Compute the area under the precision-recall curve
    auprc = auc(recall, precision)
    
    return auprc

def plugin_RCE_est(correctness, uncertainties, num_bins=20, p=1, use_kernel = False, sigma=0.1, **kwargs):
    '''
    Input:
        uncertainties: (U_1, ... , U_n) \in R^n
        correctness: (A_1, ..., A_n) \in [0, 1]^n
        num_bins: B
    Output:
        Plug-in estimator of l_p-ERCE(f)^p w.r.t. B equal-mass bins
    '''

    n = len(correctness)
    regressed_correctness, uncertainty_cdfs = regressed_correctness_vs_uncertainty_cdf(correctness=correctness, uncertainties=uncertainties,\
                                                                                        num_bins=num_bins, use_kernel_regress=use_kernel, sigma=sigma)
    regressed_correctness_inv_cdfs = np.array([(np.sum([regressed_correctness[i] <= regressed_correctness])-1)/(n-1) for i in range(n)])
    if not use_kernel:
        # compute the detied (due to binning) inverse cdf of regressed correctness
        regressed_correctness_detied_inv_cdfs = np.zeros(n)
        bin_endpoints = [round(ele) for ele in np.linspace(0, n, num_bins+1)]
        for idx_bin in range(1, num_bins+1):
            lo, hi = bin_endpoints[idx_bin-1], bin_endpoints[idx_bin]
            if hi > lo:
                regressed_correctness_detied_inv_cdfs[lo:hi] = np.mean(regressed_correctness_inv_cdfs[lo:hi]) - (hi-lo-1)/(2*(n-1))
    if use_kernel:
        regressed_correctness_detied_inv_cdfs = regressed_correctness_inv_cdfs

    if p == 1:
        return np.sum(np.abs(regressed_correctness_detied_inv_cdfs - uncertainty_cdfs))/n
    elif p == 2:
        return np.sum((regressed_correctness_detied_inv_cdfs - uncertainty_cdfs)**2)/n
    else:
        raise ValueError("Please specify a valid order p!")

# def plugin_erce_est(uncertainties, correctness, num_bins=20, p=1, use_kernel = False, sigma=0.1, **kwargs):
#     '''
#     Input:
#         uncertainties: (U_1, ... , U_n) \in R^n
#         correctness: (A_1, ..., A_n) \in [0, 1]^n
#         num_bins: B
#     Output:
#         Plug-in estimator of l_p-ERCE(f)^p w.r.t. B equal-mass bins
#     '''
#     n = len(correctness)
#     bin_endpoints = [round(ele) for ele in np.linspace(0, n, num_bins+1)]
#     sorted_indices = np.argsort(uncertainties)
#     correctness = correctness[sorted_indices]
#     uncertainties = uncertainties[sorted_indices]
#     a_map = -np.ones(n)
#     u_map = np.zeros(n)
#     a_hats = []
#     u_hats = []
#     # compute a_hat, u_hat and a_map: i -> a_hat, u_map: i -> u_hat
#     for idx_bin in range(1, num_bins+1):
#         lo, hi = bin_endpoints[idx_bin-1], bin_endpoints[idx_bin]
#         if hi > lo:
#             # if idx_bin == num_bins:  hi += 1
#             bin_correctness = correctness[lo:hi]
#             a_hat = np.mean(bin_correctness)
#             a_map[lo:hi] = a_hat
#             u_hat = np.mean(uncertainties[lo:hi]) if hi > lo else None
#             u_map[lo:hi] = u_hat
#             a_hats.append(a_hat)
#             u_hats.append(u_hat)
#     result = 0
#     for a_hat, u_hat in zip(a_hats, u_hats):

#         count_correnct = (np.sum(a_map >= a_hat) - (1+np.sum(a_map == a_hat)) / 2) / (n-1)
#         count_uncertainty = (np.sum(u_map <= u_hat) - (1+np.sum(u_map == u_hat)) / 2) / (n-1)
#         if p == 1:
#             result += np.abs(count_correnct - count_uncertainty) * np.sum(u_map == u_hat) / n
#         elif p == 2:
#             result += (count_correnct - count_uncertainty) ** 2 * np.sum(u_map == u_hat) / n
#         else:
#             raise ValueError("Please specify a valid order p!")
#     return result

# def rank_erce_est(uncertainties, correctness, num_bins=20, p=1):
#     '''
#     Input:
#         uncertainties: (U_1, ... , U_n) \in R^n
#         correctness: (A_1, ..., A_n) \in [0, 1]^n
#         num_bins: B
#     Output:
#         Plug-in estimator of l_p-ERCE(f)^p w.r.t. B equal-mass bins
#     '''
#     n = len(correctness)
#     bin_endpoints = [round(ele) for ele in np.linspace(0, n, num_bins+1)]
#     sorted_indices = np.argsort(uncertainties)
#     correctness = correctness[sorted_indices]
#     uncertainties = uncertainties[sorted_indices]
#     a_rank_map = np.zeros(num_bins)
#     u_rank_map = np.zeros(num_bins)
#     num_count = np.zeros(num_bins)
#     # compute cdf of correctness
#     correct_ranks, uncertainty_ranks = np.zeros(n), np.zeros(n)
#     for i in range(n):
#         correct_ranks[i] = (np.sum(correctness[i] >= correctness)-1) / (n-1)
#         uncertainty_ranks[i] = (np.sum(uncertainties[i] >= uncertainties)-1) / (n-1)
#     # compute a_hat, u_hat and a_map: i -> a_hat, u_map: i -> u_hat
#     for idx_bin in range(1, num_bins+1):
#         lo, hi = bin_endpoints[idx_bin-1], bin_endpoints[idx_bin]
#         if hi > lo:
#             bin_correct_ranks = correct_ranks[lo:hi]
#             a_rank_hat = np.mean(bin_correct_ranks)
#             a_rank_map[idx_bin-1] = a_rank_hat
#             u_rank_hat = np.mean(uncertainty_ranks[lo:hi])
#             u_rank_map[idx_bin-1] = u_rank_hat
#         num_count[idx_bin-1] = hi-lo

#     result = 0
#     for idx in range(num_bins):
#         tmp = a_rank_map[idx] - (1-u_rank_map[idx])
#         if p == 1:
#             result += np.abs(tmp) * num_count[idx] / n
#         elif p == 2:     
#             result += tmp ** 2 * num_count[idx] / n
#         else:
#             raise ValueError("Please specify a valid order p!")
#     return result

# def debias_rank_erce_est(uncertainties, correctness, num_bins=20):
#     '''
#     Input:
#         uncertainties: (U_1, ... , U_n) \in R^n
#         correctness: (A_1, ..., A_n) \in [0, 1]^n
#         num_bins: B
#     Output:
#         Plug-in estimator of l_2-Rank-ERCE(f)^2 w.r.t. B equal-mass bins
#     '''
#     n = len(correctness)
#     bin_endpoints = [round(ele) for ele in np.linspace(0, n, num_bins+1)]
#     sorted_indices = np.argsort(uncertainties)
#     correctness = correctness[sorted_indices]
#     uncertainties = uncertainties[sorted_indices]
#     # compute cdf of correctness
#     correct_ranks, uncertainty_ranks = np.zeros(n), np.zeros(n)
#     for i in range(n):
#         correct_ranks[i] = (np.sum(correctness[i] >= correctness)-1)/(n-1)
#         # breakpoint()
#         uncertainty_ranks[i] = (np.sum(uncertainties[i] >= uncertainties)-1)/(n-1)
#     # breakpoint()
#     result = 0
#     for idx_bin in range(1, num_bins+1):
#         lo, hi = bin_endpoints[idx_bin-1], bin_endpoints[idx_bin]
#         for i in range(lo, hi):
#             for j in range(i+1, hi):
#                 correct_rank_i, correct_rank_j \
#                     = correct_ranks[i]+(correct_ranks[i]-(correctness[i]<=correctness[j]))/(n-2),\
#                     correct_ranks[j]+(correct_ranks[j]-(correctness[j]<=correctness[i]))/(n-2)
#                 uncertainty_rank_i, uncertainty_rank_j \
#                     = uncertainty_ranks[i]+(uncertainty_ranks[i]-(uncertainties[i]<=uncertainties[j]))/(n-2),\
#                     uncertainty_ranks[j]+(uncertainty_ranks[j]-(uncertainties[j]<=uncertainties[i]))/(n-2)
#                 result += 2*(correct_rank_i-1+uncertainty_rank_i)*(correct_rank_j-1+uncertainty_rank_j)/((n-1)*(hi-lo))
#     return result
    
# def adaptive_rank_erce_est(uncertainties, correctness):
#     '''
#     Input:
#         uncertainties: (U_1, ... , U_n) \in R^n
#         correctness: (A_1, ..., A_n) \in [0, 1]^n
#         num_bins: B
#     Output:
#         Plug-in estimator of l_2-Rank-ERCE(f)^2 w.r.t. B equal-mass bins
#     '''
#     n = len(correctness)
#     num_bins_list = [2**b for b in range(2, np.floor(np.log2(n)-2).astype(int))]
#     if not num_bins_list:
#         raise ValueError("The evaluation dataset is too small!")
#     return np.max([debias_rank_erce_est(uncertainties, correctness, num_bins) for num_bins in num_bins_list])

# def nested_rank_erce_est(uncertainties, correctness, num_bins=20, p=1):
#     '''
#     Input:
#         uncertainties: (U_1, ... , U_n) \in R^n
#         correctness: (A_1, ..., A_n) \in [0, 1]^n
#         num_bins: B
#     Output:
#         Plug-in estimator of l_p-ERCE(f)^p w.r.t. B equal-mass bins
#     '''
#     n = len(correctness)
#     bin_endpoints = [round(ele) for ele in np.linspace(0, n, num_bins+1)]
#     sorted_indices = np.argsort(uncertainties)
#     correctness = correctness[sorted_indices]
#     uncertainties = uncertainties[sorted_indices]
#     a_map = np.zeros(num_bins)
#     u_map = np.zeros(num_bins)
#     num_count = np.zeros(num_bins)
#     # compute cdf of correctness
#     correct_ranks, uncertainty_ranks = np.zeros(n), np.zeros(n)
#     for i in range(n):
#         correct_ranks[i] = (np.sum(correctness[i] >= correctness)-1) / (n-1)
#         uncertainty_ranks[i] = (np.sum(uncertainties[i] >= uncertainties)-1) / (n-1)
#     for idx_bin in range(1, num_bins+1):
#         lo, hi = bin_endpoints[idx_bin-1], bin_endpoints[idx_bin]
#         if hi > lo:
#             bin_correct_ranks = correct_ranks[lo:hi]
#             a_hat = np.mean(bin_correct_ranks)
#             a_map[idx_bin-1] = a_hat
#             u_hat = np.mean(uncertainty_ranks[lo:hi])
#             u_map[idx_bin-1] = u_hat
#         num_count[idx_bin-1] = hi-lo

#     a_map_ranks = np.zeros(num_bins)
#     u_map_ranks = np.zeros(num_bins)
#     for idx in range(num_bins):
#         a_map_ranks[idx] = (np.sum(a_map[idx] >= a_map)-1) / (num_bins-1)
#         u_map_ranks[idx] = (np.sum(u_map[idx] >= u_map)-1) / (num_bins-1)

#     result = 0
#     for idx in range(num_bins):
#         tmp = a_map_ranks[idx] - (1-u_map_ranks[idx])
#         if p == 1:
#             result += np.abs(tmp) * num_count[idx] / n
#         elif p == 2:     
#             result += tmp ** 2 * num_count[idx] / n
#         else:
#             raise ValueError("Please specify a valid order p!")
#     return result

def reflected_Gaussian_kernel(x, y, sigma = 0.05):
    '''
    Compute the reflected Gaussian kernel between two numbers in [0,1].
    Input:
        x \in [0, 1]
        y \in [0, 1]
        sigma: the kernel width
    Output:
        the paired kernel values K_\sigma(x, y) \in R_+
    '''
    return np.sum([norm.pdf(x-y+2*k, 0, sigma)+norm.pdf(y-x+2*k, 0, sigma) for k in range(-4, 4)]) 

def regressed_correctness_vs_uncertainty_cdf(correctness, uncertainties, num_bins = 20, use_kernel_regress = False, sigma = 0.1):
    '''
    Compute the regressed correctness levels with binning or kernel smoothing.
    Input:
        correctness: (a_1, ..., a_n) \in R^n 
        uncertainties: (u_1, ..., u_n ) \in R^n
    Output:
        uncertainty_cdfs: (p_1=0, ..., p_n=1) \in R^n the CDF estimates of sorted uncertainties
        regressed_correctness: (\bar{a}_1, ..., \bar{a}_n) \in R^n the regressed correctness 
            listed in the sorted order
    '''
    n = len(uncertainties)
    sorted_indices = np.argsort(uncertainties)
    sorted_correctness = correctness[sorted_indices]
    uncertainty_cdfs =  np.arange(0, n)/ (n-1)
    regressed_correctness = np.zeros(n)
    if not use_kernel_regress:
        bin_endpoints = [round(ele) for ele in np.linspace(0, n, num_bins+1)]
        for idx_bin in range(1, num_bins+1):
            lo, hi = bin_endpoints[idx_bin-1], bin_endpoints[idx_bin]
            if hi > lo:
                a_hat = np.mean(sorted_correctness[lo:hi])
                for i in range(lo, hi):
                    regressed_correctness[i] = a_hat
        return regressed_correctness, uncertainty_cdfs
    elif use_kernel_regress:
        kernel_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                kernel_mat[i][j] = reflected_Gaussian_kernel(uncertainty_cdfs[i], uncertainty_cdfs[j], sigma)
                kernel_mat[j][i] = kernel_mat[i][j]
        regressed_correctness = kernel_mat@sorted_correctness/kernel_mat.sum(axis=1)
        return regressed_correctness, uncertainty_cdfs
    
def correctness_variability_vs_uncertainty_cdf(correctness, uncertainties, num_bins = 20):
    '''
    Compute the correctness variability levels with binning or kernel smoothing.
    Input:
        correctness: (a_1, ..., a_n) \in R^n 
        uncertainties: (u_1, ..., u_n ) \in R^n
    Output:
        correctness_bin_means: (\bar{a}_1, ..., \bar{a}_B) \in R^B the regressed correctness 
            listed for each bin
        correctness_bin_stds: (s_1, ..., s_B) \in R^B the standard deviation of correctness 
            listed for each bin
    '''
    n = len(uncertainties)
    sorted_indices = np.argsort(uncertainties)
    sorted_correctness = correctness[sorted_indices]
    correctness_bin_means = np.zeros(num_bins)
    correctness_bin_stds = np.zeros(num_bins)
    bin_endpoints = [round(ele) for ele in np.linspace(0, n, num_bins+1)]
    for idx_bin in range(1, num_bins+1):
        lo, hi = bin_endpoints[idx_bin-1], bin_endpoints[idx_bin]
        if hi > lo:
            correctness_bin_means[idx_bin-1] = np.mean(sorted_correctness[lo:hi])
            correctness_bin_stds[idx_bin-1] = np.std(sorted_correctness[lo:hi])
    return correctness_bin_means, correctness_bin_stds

def histogram_binning(correctness, uncertainties, num_bins = 20):
    '''
    Compute the histogram binning for re-rank-calibration with the calibratition set.
    Input:
        correctness: (a_1, ..., a_n) \in R^n 
        uncertainties: (u_1, ..., u_n ) \in R^n
    Output:
        bin_boundaries: (b_1, ..., b_{B-1}) \in R^{B} the bin boundaries of sorted uncertainties
        bin_correctness: (\bar{a}_1, ..., \bar{a}_B) \in R^B the regressed correctness 
            for each bin
    '''
    n = len(uncertainties)
    sorted_indices = np.argsort(uncertainties)
    sorted_correctness = correctness[sorted_indices]
    sorted_uncertainties = uncertainties[sorted_indices]
    bin_correctness = np.zeros(num_bins)

    bin_endpoints = [round(ele) for ele in np.linspace(0, n, num_bins+1)]
    bin_boundaries = sorted_uncertainties[bin_endpoints[1:-1]]
    for idx_bin in range(1, num_bins+1):
        lo, hi = bin_endpoints[idx_bin-1], bin_endpoints[idx_bin]
        a_hat = np.mean(sorted_correctness[lo:hi]) if hi > lo else 0
        bin_correctness[idx_bin-1] = a_hat
    return bin_boundaries, bin_correctness

def histogram_recalibration(correctness, uncertainties, num_bins = 20, split_ratio = 0.5):
    '''
    Recalibrate with histogram binning.
    Input:
        correctness: (a_1, ..., a_n) \in R^n 
        uncertainties: (u_1, ..., u_n ) \in R^n
        split_ratio: the proportion of calibration set
    Output:
        test_correctness: (a_1, ..., a_{m}) \in R^{m} correctness of the remaining test set where m = n*(1-split_ratio)
        test_uncertainties: (\hat{u}_1, ..., \hat{u}_m) \in R^m the recalibrated uncertainties
    '''
    np.random.seed(2024) # for reproducibility

    n = len(uncertainties)
    cal_size = round(n*split_ratio)
    reorder = np.random.permutation(range(n))
    correctness, uncertainties = correctness[reorder], uncertainties[reorder]+1e-3*np.random.randn(n) # break ties in binning
    cal_correctness, test_correctness = correctness[:cal_size], correctness[cal_size:]
    cal_uncertainties, test_uncertainties = uncertainties[:cal_size], uncertainties[cal_size:]
    bin_boundaries, bin_correctness = histogram_binning(cal_correctness, cal_uncertainties, num_bins)

    for idx_u, u in enumerate(test_uncertainties):
        idx_bin = bisect.bisect_left(bin_boundaries, u)
        test_uncertainties[idx_u] = bin_correctness[idx_bin]
        
    return test_correctness, test_uncertainties