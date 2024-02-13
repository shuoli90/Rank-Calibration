import os
import sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from metrics import calibration

if __name__ == '__main__':
    correctness = np.random.uniform(0, 1, 100)
    confidences = np.random.uniform(0, 1, 100)
    result = calibration.plugin_erce_est(1-correctness, correctness, num_bins=20, p=1)
    print('Best case', result)
    result = calibration.plugin_erce_est(1/correctness, correctness, num_bins=20, p=1)
    print('Best case 2', result)
    result = calibration.plugin_erce_est(1-correctness**2, correctness, num_bins=20, p=1)
    print('Best case 3', result)
    result = calibration.plugin_erce_est(1/correctness**2, correctness, num_bins=20, p=1)
    print('Best case 4', result)


    result = calibration.plugin_erce_est(correctness, correctness, num_bins=20, p=1)
    print('Worst case', result)
    result = calibration.plugin_erce_est(confidences, correctness, num_bins=20, p=1)
    print('Independent case', result)

    result = calibration.plugin_erce_est(1/correctness+0.5 * np.random.randn(100), correctness, num_bins=20, p=1)
    print('Noisy case', result)