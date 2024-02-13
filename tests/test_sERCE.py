import os
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from metrics import calibration

if __name__ == '__main__':
    # correctness = np.random.uniform(0, 1, 100)
    # confidences = np.random.uniform(0, 1, 100)
    # result = calibration.plugin_erce_est(1-correctness, correctness, num_bins=20, p=1)
    # print('Best case', result)
    # result = calibration.plugin_erce_est(1/correctness, correctness, num_bins=20, p=1)
    # print('Best case 2', result)
    # result = calibration.plugin_erce_est(1-correctness**2, correctness, num_bins=20, p=1)
    # print('Best case 3', result)
    # result = calibration.plugin_erce_est(1/correctness**2, correctness, num_bins=20, p=1)
    # print('Best case 4', result)


    # result = calibration.plugin_erce_est(correctness, correctness, num_bins=20, p=1)
    # print('Worst case', result)
    # result = calibration.plugin_erce_est(confidences, correctness, num_bins=20, p=1)
    # print('Independent case', result)

    # result = calibration.plugin_erce_est(1/correctness+0.5 * np.random.randn(100), correctness, num_bins=20, p=1)
    # print('Noisy case', result)

    file_names = [file for file in os.listdir("../tmp") if file.endswith('.csv')]
    bins = 20
    for file_name in file_names:
        dir = os.path.join("../tmp", file_name)
        df = pd.read_csv(dir).dropna(axis=0)
        correctness = df['score'].to_numpy()
        if len(correctness) < bins:
            continue
        if 'semantic_entropy' in file_name:
            confidence = -df['confidence'].to_numpy()
        else:
            confidence = df['confidence'].to_numpy()
        result = calibration.plugin_erce_est(confidence, correctness, num_bins=bins, p=1)
        print(f"File: {file_name}, ERCE: {result}")
    
    correctness = np.random.uniform(0, 1, len(correctness))
    uncertainties = np.random.uniform(0, 1, len(correctness))
    result = calibration.plugin_erce_est(correctness, uncertainties, num_bins=bins, p=1)
    print('Worst case', result)