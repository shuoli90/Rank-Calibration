import os
import sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from metrics import calibration

if __name__ == '__main__':
    correctness = np.random.uniform(0, 1, 1000)
    confidences = np.random.uniform(0, 1, 1000)
    result = calibration.sERCE(correctness, 1-correctness, bins=100, p=1)
    print('Best case', result)
    result = calibration.sERCE(correctness, confidences, bins=100, p=1)
    print('Worst case', result)
