import os
import sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from metrics import calibration

if __name__ == '__main__':
    correctness = np.random.rand(1000)
    confidences = np.random.rand(1000)
    result = calibration.sERCE(correctness, 1-correctness, bins=10, p=1)
