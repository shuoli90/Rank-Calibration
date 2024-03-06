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
    # fig, ax = plt.subplots()
    # uncertainties = np.random.rand(5000)
    # correctness = 1-uncertainties+0.3*np.random.randn(5000)
    # # regressed_correctness, uncertainty_cdfs = make_plots.regressed_correctness_vs_uncertainty_cdf(correctness, uncertainties, num_bins = 20, use_kernel_regress = True, sigma = 0.1)
    # # breakpoint()
    # ax = make_plots.indication_diagram(correctness, uncertainties, fig, ax)
    # plt.savefig(f'tests/test_tmp_indication_diagram.png')

    # Assuming you have the following data
    B = 20
    ucc = np.linspace(0.03, 0.97, B)  # Replace this with your confidence intervals
    acc = np.clip(1-ucc+0.2*(2*np.random.rand(B)-1), 0, 1)  # Replace this with your accuracy measurements
    # lo, hi = np.min(u, a), np.max(u, a)

    # Create figure and axes
    fig, ax = plt.subplots()

    # Plot the bars for empirical accuracy
    # breakpoint()
    ax.bar(np.arange(B)/B*100, np.minimum(1-ucc, acc)*100, width=100/B, color='crimson', align='edge', label='CDF(E[A|U])')
    ax.bar(np.arange(B)/B*100, (1-ucc-np.minimum(1-ucc, acc))*100, width=100/B, bottom=np.minimum(1-ucc, acc)*100, color='dodgerblue', align='edge', label='CDF(U)')
    ax.bar(np.arange(B)/B*100, (acc-np.minimum(1-ucc, acc))*100, width=100/B, bottom=np.minimum(1-ucc, acc)*100, color='salmon', align='edge')
    
    # Plot the diagonal line

    ax.plot([100, 0], [0, 100], linestyle='--', color='black', linewidth=2)

    # Add legend
    ax.legend(loc='upper right', frameon=False, fontsize=20)

    # Add labels and title
    # ax.set_xlabel('Predicted Confidence')
    # ax.set_ylabel('Empirical Accuracy')
    ax.set_title('Indication Diagram')

    # Set the limits of the plot
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

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