import numpy as np
from scipy.stats import ks_2samp
from calibration import plugin_ece_est

def consistency_resampling(scores):
  '''
  Input:
    scores: (Z_1, ... , Z_n) \in [0, 1]^n
  Output:
    Sample (Z_1', ... , Z_n') with replacement from {Z_1, ... , Z_n}
    Output (Y_1', ... , Y_n') where Y_i' ~ Ber(Z_i')
  '''

  n = len(scores)
  sampled_scores = np.random.choice(scores, size=n, replace=True)
  sampled_labels = np.random.binomial(1, sampled_scores, n)

  return sampled_scores, sampled_labels

def adaptive_T_Cal(scores, labels, alpha=0.05, MC_trials=3000):
  '''
  Input:
    scores: (Z_1, ... , Z_n) \in [0, 1]^n
    labels: (Y_1, ... , Y_n) \in {0, 1}^n
    alpha: Size of test (type I error, false detection rate)
  Output:
    Result of adaptive T-Cal test (Lee et al.) to exmaine a model
    is perfectly calibrated or not.
    Return True if the null hypothesis of perfect calibration is rejected
  '''
  
  n = len(labels)
  B = int(2 * np.log2(n / np.sqrt(np.log(n))))

  for b in range(1, B + 1): 
    num_bins = 2**b
    MC_dpe = np.zeros(MC_trials,)
    for t in range(MC_trials):
      MC_scores, MC_labels = consistency_resampling(scores)
      MC_dpe[t] = plugin_ece_est(MC_scores, MC_labels, num_bins, 2, True)

    test_dpe = plugin_ece_est(scores, labels, num_bins, 2, True)
    threshhold = np.quantile(MC_dpe, 1 - alpha/B)
    
    if test_dpe > threshhold:
      return True

  return False

class post_hoc_test():

    def __init__(self, test_name='KS'):
        self.test_name = test_name
        if test_name not in ['KS', 'T-Cal']:
           raise ValueError("Please specify a valid test!")

    def __call__(self, scores, labels, alpha=0.05, MC_trials=3000):
        if self.test_name == 'KS':
            # to examine the distributional difference
            data1, data2 = scores[labels==0], scores[labels==1]
            result = ks_2samp(data1, data2)
            return {'P-value': result.pvalue, 'stat': result.statistic}
        
        elif self.test_name == 'T-Cal':
            # to examine the perfect calibration
            result = adaptive_T_Cal(scores, labels, alpha, MC_trials)
            return result