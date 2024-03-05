import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from indicators import blackbox
from models import opensource, gpt
from utils import text_processing
from utils import make_plots
import numpy as np
import matplotlib.pyplot as plt

import relplot as rp

if __name__ == '__main__':
    fig, ax = plt.subplots()
    uncertainties = np.random.rand(5000)
    correctness = 1-uncertainties+0.3*np.random.randn(5000)
    # regressed_correctness, uncertainty_cdfs = make_plots.regressed_correctness_vs_uncertainty_cdf(correctness, uncertainties, num_bins = 20, use_kernel_regress = True, sigma = 0.1)
    # breakpoint()
    ax = make_plots.indication_diagram(correctness, uncertainties, fig, ax)
    plt.savefig(f'tests/test_tmp_indication_diagram.png')
    # N = 5000
    # f = np.random.rand(N)
    # y = (np.random.rand(N) > 1-(f + 0.2*np.sin(2*np.pi*f)))*1

    # ## compute calibration error (smECE) and plot
    # print('calibration error:', rp.smECE(f, y))
    # diagram = rp.prepare_rel_diagram(f, y) # compute calibration data (dictionary)
    # breakpoint()
    # print('calibration error:', diagram['ce']) 
    # plt.plot(diagram['mesh'], diagram['mu']) # plot the calibration curve manually
    # fig, ax = rp.plot_rel_diagram(diagram) # plot the diagram in a new figure
    # # fig, ax = rp.rel_diagram(f, y)
    # # fig.show()