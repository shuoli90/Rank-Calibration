import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from metrics.calibration import reflected_Gaussian_kernel, regressed_correctness_vs_uncertainty_cdf
import matplotlib.ticker as mtick
from metrics.calibration import reflected_Gaussian_kernel, regressed_correctness_vs_uncertainty_cdf, AUARC
import matplotlib.ticker as mtick

def AUROC_vs_Correctness(correctness, confidence, thresholds, ax, plot=True, **kwargs):
    # compute AUROC for different correctness thresholds
    aurocs = []
    for threshold in thresholds:
        y_true = correctness >= threshold
        y_score = confidence
        try:
            auroc = roc_auc_score(y_true, y_score)
            aurocs.append(auroc)
        except ValueError:
            print('problematic')
        #     raise ValueError(f"Threshold {threshold} has no positive samples")
    # plot
    if plot:
        df = pd.DataFrame(dict(AUROC=aurocs, Threshold=thresholds))
        sns.lineplot(x="Threshold", y="AUROC", data=df, ax=ax, **kwargs)
        return ax
    else:
        return aurocs

def AUROC_vs_Correctness_average(correctnesses, confidences, thresholds, ax, plot=True, **kwargs):
    # compute AUROC for different correctness thresholds
    aurocs = []
    for threshold in thresholds:
        aurocs_tmp = []
        for idx in range(correctnesses.shape[1]):
            correctness = correctnesses[:, idx]
            confidence = confidences[:, idx]
            y_true = correctness >= threshold
            y_score = confidence
            try:
                auroc = roc_auc_score(y_true, y_score)
                aurocs_tmp.append(auroc)
            except ValueError:
                raise ValueError(f"Problematic")
        aurocs.append(np.mean(aurocs_tmp))
    # plot
    if plot:
        df = pd.DataFrame(dict(AUROC=aurocs, Threshold=thresholds))
        sns.lineplot(x="Threshold", y="AUROC", data=df, ax=ax, **kwargs)
        return ax
    else:
        return aurocs

def AUARC_vs_Correctness(correctness, confidence, thresholds, ax, plot=True, **kwargs):
    # compute AUROC for different correctness thresholds
    aurocs = []
    for threshold in thresholds:
        y_true = correctness >= threshold
        y_score = confidence
        try:
            auroc = AUARC(y_true, y_score)
            aurocs.append(auroc)
        except ValueError:
            print('problematic')
        #     raise ValueError(f"Threshold {threshold} has no positive samples")
    # plot
    if plot:
        df = pd.DataFrame(dict(AUROC=aurocs, Threshold=thresholds))
        sns.lineplot(x="Threshold", y="AUARC", data=df, ax=ax, **kwargs)
        return ax
    else:
        return aurocs

def AUARC_vs_Correctness_average(correctnesses, confidences, thresholds, ax, plot=True, **kwargs):
    # compute AUROC for different correctness thresholds
    aurocs = []
    for threshold in thresholds:
        aurocs_tmp = []
        for idx in range(correctnesses.shape[1]):
            correctness = correctnesses[:, idx]
            confidence = confidences[:, idx]
            y_true = correctness >= threshold
            y_score = confidence
            try:
                auroc = AUARC(y_true, y_score)
                aurocs_tmp.append(auroc)
            except ValueError:
                raise ValueError(f"Problematic")
        aurocs.append(np.mean(aurocs_tmp))
    # plot
    if plot:
        df = pd.DataFrame(dict(AUROC=aurocs, Threshold=thresholds))
        sns.lineplot(x="Threshold", y="AUARC", data=df, ax=ax, **kwargs)
        return ax
    else:
        return aurocs

def indication_diagram(correctness, uncertainties, fig, ax, num_bins=20, use_kernel = False, sigma=0.1, **kwargs):
    '''
        Draw the indication diagram.
        Option I: using equal-width binning.
        Option II: using kernel smoothing.
    '''
    n = len(correctness)
    regressed_correctness, uncertainty_cdfs = regressed_correctness_vs_uncertainty_cdf(correctness=correctness, uncertainties=uncertainties,\
                                                                                        num_bins=num_bins, use_kernel_regress=use_kernel, sigma=sigma)
    
    regressed_correctness_cdfs = np.array([(np.sum([regressed_correctness[i] >= regressed_correctness])-1)/(n-1) for i in range(n)])
    if not use_kernel:
        bin_endpoints = [round(ele) for ele in np.linspace(0, n, num_bins+1)]
        # compute binned a_hat, u_hat for histogram ploting 
        a_hats, u_hats = [], []
        for idx_bin in range(1, num_bins+1):
            # breakpoint()
            lo, hi = bin_endpoints[idx_bin-1], bin_endpoints[idx_bin]
            if hi > lo:
                a_hat = np.mean(regressed_correctness_cdfs[lo:hi]) - (hi-lo-1)/(2*(n-1))
                u_hat = np.mean(uncertainty_cdfs[lo:hi])
                a_hats.append(a_hat)
                u_hats.append(u_hat)
        # sns.barplot(x=[round(u_hat*100) for u_hat in u_hats], y=[a_hat*100 for a_hat in a_hats], ax=ax, **kwargs)
        ucc, acc, B = np.array(u_hats), np.array(a_hats), num_bins
        ax.bar(np.arange(B)/B*100, np.minimum(1-ucc, acc)*100, width=100/B, color='crimson', align='edge', label='CDF($\mathbb{E}[A|U]$)')
        ax.bar(np.arange(B)/B*100, (1-ucc-np.minimum(1-ucc, acc))*100, width=100/B, bottom=np.minimum(1-ucc, acc)*100, color='dodgerblue', align='edge', label='CDF($U$)')
        ax.bar(np.arange(B)/B*100, (acc-np.minimum(1-ucc, acc))*100, width=100/B, bottom=np.minimum(1-ucc, acc)*100, color='salmon', align='edge')
        # Plot the anti-diagonal line
        ax.plot([100, 0], [0, 100], linestyle='--', color='black', linewidth=2)
        # Add legend
        ax.legend(loc='upper right', frameon=False, fontsize=15)

        ax.set_xlabel('Percentage of Unertainty (%)', fontsize=15)
        ax.set_ylabel('Percentage of Regressed Correctness (%)', fontsize=15)
        plt.xlim(0, 100)
        ax.grid()
        fig.tight_layout()
        return ax
    else:
        plt.plot(uncertainty_cdfs*100, regressed_correctness_cdfs*100, color='r', linewidth=2)
        ax.set_xlabel('Percentage of Unertainty (%)', fontsize=15)
        ax.set_ylabel('Percentage of Regressed Correctness (%)', fontsize=15)
        ax.grid()
        fig.tight_layout()
        return ax