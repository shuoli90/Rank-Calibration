import os
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib.pyplot as plt
from metrics import calibration
from utils import make_plots
from matplotlib.ticker import FormatStrFormatter


if __name__ == '__main__':
    # print('Best case:\n')
    # correctness = np.random.uniform(0, 1, 5000)
    # # confidences = np.random.uniform(0, 1, 200)
    # noisy_correctness = correctness
    # # +0.5 * np.random.randn(len(correctness))
    
    # result = calibration.plugin_erce_est(1-correctness, noisy_correctness, num_bins=20, p=1)
    # print('plugin-erce-1: ', result)

    # result = calibration.plugin_erce_est(1-correctness, noisy_correctness, num_bins=20, p=2)
    # print('plugin-erce-2: ', result)

    # result = calibration.rank_erce_est(1-correctness,noisy_correctness, num_bins=20, p=1)
    # print('plugin-rank-erce-1: ', result)

    # result = calibration.rank_erce_est(1-correctness, noisy_correctness, num_bins=20, p=2)
    # print('plugin-rank-erce-2: ', result)

    # result = calibration.debias_rank_erce_est(1-correctness, noisy_correctness, num_bins=20)
    # print('plugin-rank-erce-2-debiased: ', result)

    # result = calibration.adaptive_rank_erce_est(1-correctness, noisy_correctness)
    # print('adaptive-rank-erce-2-debiased: ', result)



    # print('Worst case:\n')
    # # correctness = np.random.uniform(0, 1, 100)
    
    # result = calibration.plugin_erce_est(correctness, noisy_correctness, num_bins=20, p=1)
    # print('plugin-erce-1: ', result)

    # result = calibration.plugin_erce_est(correctness, noisy_correctness, num_bins=20, p=2)
    # print('plugin-erce-2: ', result)

    # result = calibration.rank_erce_est(correctness, noisy_correctness, num_bins=20, p=1)
    # print('plugin-rank-erce-1: ', result)

    # result = calibration.rank_erce_est(correctness, noisy_correctness, num_bins=20, p=2)
    # print('plugin-rank-erce-2: ', result)

    # result = calibration.debias_rank_erce_est(correctness, noisy_correctness, num_bins=20)
    # print('plugin-rank-erce-2-debiased: ', result)

    # result = calibration.adaptive_rank_erce_est(correctness, noisy_correctness)
    # print('adaptive-rank-erce-2-debiased: ', result)

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

    # print('Original')
    # file_names = [file for file in os.listdir("tmp") if file.endswith('.csv')]
    bins = 20
    # for file_name in file_names:
    #     dir = os.path.join("../tmp", file_name)
    #     df = pd.read_csv(dir).dropna(axis=0)
    #     correctness = df['score'].to_numpy()
    #     # if len(correctness) >= bins:
    #         # continue
    #     if 'semantic_entropy' in file_name:
    #         confidence = -df['confidence'].to_numpy()
    #     else:
    #         confidence = df['confidence'].to_numpy()
    #     result = calibration.plugin_erce_est(confidence, correctness, num_bins=bins, p=1)
    #     print(f"File: {file_name}, ERCE: {result}")
    
    # correctness = np.random.uniform(0, 1, len(correctness))+0.2 * np.random.randn(100)
    # uncertainties = np.random.uniform(0, 1, len(correctness))
    # result = calibration.plugin_erce_est(uncertainties, correctness, num_bins=bins, p=1)
    # print('Worst case', result)

    # print('Rank-based')
    # file_names = [file for file in os.listdir("tmp") if file.endswith('.csv')]
    # bins = 20
    # for file_name in file_names:
    #     dir = os.path.join("../tmp", file_name)
    #     df = pd.read_csv(dir).dropna(axis=0)
    #     correctness = df['score'].to_numpy()
    #     # if len(correctness) >= bins:
    #         # continue
    #     if 'semantic_entropy' in file_name:
    #         confidence = -df['confidence'].to_numpy()
    #     else:
    #         confidence = df['confidence'].to_numpy()
    #     result = calibration.rank_erce_est(confidence, correctness, num_bins=bins, p=1)
    #     print(f"File: {file_name}, ERCE: {result}")
    
    # correctness = np.random.uniform(0, 1, len(correctness))
    # uncertainties = np.random.uniform(0, 1, len(correctness))
    # result = calibration.rank_erce_est(uncertainties, correctness, num_bins=bins, p=1)
    # print('Worst case', result)


    # fig, ax = plt.subplots()
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # ax = make_plots.histogram(correctness, uncertainties, fig, ax)
    # plt.savefig('histogram.png')

    # print('debiased_new')
    # file_names = [file for file in os.listdir("tmp") if file.endswith('.csv')]
    # bins = 20
    # for file_name in file_names:
    #     dir = os.path.join("../tmp", file_name)
    #     df = pd.read_csv(dir).dropna(axis=0)
    #     correctness = df['score'].to_numpy()
    #     # if len(correctness) >= bins:
    #         # continue
    #     if 'semantic_entropy' in file_name:
    #         confidence = -df['confidence'].to_numpy()
    #     else:
    #         confidence = df['confidence'].to_numpy()
    #     result = calibration.rank_erce_est(confidence, correctness, num_bins=bins, p=1)
    #     print(f"File: {file_name}, ERCE: {result}")
    
    # correctness = np.random.uniform(0, 1, len(correctness))
    # uncertainties = np.random.uniform(0, 1, len(correctness))
    # result = calibration.rank_erce_est(uncertainties, correctness+0.2 * np.random.randn(100), num_bins=bins, p=1)
    # print('Worst case', result)

    n = 300
    u = np.random.uniform(0.25, 0.75, size=n)
    fu = 2*u-0.5
    def perturb_dist(confidences, smoothness, half_num_waves, scale):
        '''
        This function constructs a un-calibrated model by pertubating the conditional
        expectation E[Y|C] from C. The pertubation is composed of many waves scaled from
        zeta function defined below. The resulted conditional expectation E[Y|C=c] is
        s-Holder smooth with respect to c.

        Input:
            confidences: (C_1, ... , C_n) \in [0, 1]^n
            smoothness: Holder exponent s
            half_num_waves: m
            scale: \rho

        Output:
            The groundtruth condition expectation
            (E[Y|C_1], ... , E[Y|C_n]) where E[Y|C_i] = C_i+sign(C_i-0.5)*rho*m^(-s)*zeta(2*m*C_i)
        '''

        is_inner = (0.25 <= confidences) & (confidences <= 0.75)
        zeta = lambda x: np.exp(-1/x/(1 - x))
        rescale = lambda x: (2*half_num_waves*(x - 0.25)) % 1
        signs = lambda x: -(np.floor(2*half_num_waves*(x - 0.25)) % 2)*2+1
        perturbations = is_inner* scale * half_num_waves**(-smoothness) * zeta(rescale(confidences)) * signs(confidences)

        return confidences+perturbations
    s = 1
    m = int(n**(2 / (4*s + 1)))//2
    rho =75
    perturbed_fu = perturb_dist(fu, s, m, rho)
    # a = np.random.binomial(1, 1-fu, n).astype('float')+1e-6*np.random.randn(n) # discrete label
    a = 1-fu + 1*np.random.randn(n)
    print(calibration.debias_rank_erce_est(u, a, 20))