import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from metrics import testing
from metrics.calibration import AUARC


import numpy as np

if __name__ == '__main__':
    # test = testing.post_hoc_test('T-Cal')

    C = np.random.rand(1000)
    # Y = np.array([int(np.random.rand() < c) for c in C])
    Y = np.random.randint(0, 2, size=1000)
    print(AUARC(C, Y))