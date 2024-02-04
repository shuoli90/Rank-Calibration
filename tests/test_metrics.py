import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from metrics import calibration

if __name__ == '__main__':
    confidences = [0.1, 0.2, 0.3, 0.4, 0.5]
    labels = [0, 0, 1, 1, 1]
    auarc = calibration.AUARC(confidences, labels)