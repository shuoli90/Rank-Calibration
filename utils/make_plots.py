import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

def AUROC_vs_Correctness(correctness, confidence, thresholds, ax, **kwargs):
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
            breakpoint()
        #     raise ValueError(f"Threshold {threshold} has no positive samples")
    # plot
    df = pd.DataFrame(dict(AUROC=aurocs, Correctness=thresholds))
    sns.lineplot(x="Correctness", y="AUROC", data=df, ax=ax, **kwargs)
    return ax

def AUROC_vs_Correctness_average(correctnesses, confidences, thresholds, ax, **kwargs):
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
    df = pd.DataFrame(dict(AUROC=aurocs, Correctness=thresholds))
    sns.lineplot(x="Correctness", y="AUROC", data=df, ax=ax, **kwargs)
    return ax

def histogram(correctness, uncertainties, fig, ax, num_bins=10, **kwargs):
    n = len(correctness)
    bin_endpoints = [round(ele) for ele in np.linspace(0, n, num_bins+1)]
    sorted_indices = np.argsort(uncertainties)
    correctness = correctness[sorted_indices]
    uncertainties = uncertainties[sorted_indices]
    a_map = -np.ones_like(correctness)
    u_map = np.zeros_like(uncertainties)
    a_hats = []
    u_hats = []
    # compute cdf of correctness
    correct = np.zeros_like(correctness)
    for i in range(len(correctness)):
        # correct[i] = (np.sum(correctness[i] >= correctness) - 1) / (n-1)
        correct[i] = (np.sum(correctness[i] > correctness)) / (n-1)
        if correctness[i] == 1:
            correct[i] = 1
    uncertainty = np.zeros_like(uncertainties)
    for i in range(len(uncertainties)):
        # uncertainty[i] = (np.sum(uncertainties[i] >= uncertainties)-1) / (n-1)
        # uncertainty[i] = (np.sum(uncertainties[i] > uncertainties)) / (n-1)
        uncertainty[i] = (np.sum(uncertainties[i] < uncertainties)) / (n-1)
    # compute a_hat, u_hat and a_map: i -> a_hat, u_map: i -> u_hat
    for idx_bin in range(1, num_bins+1):
        lo, hi = bin_endpoints[idx_bin-1], bin_endpoints[idx_bin]
        if hi > lo:
            bin_correctness = correct[lo:hi]
            a_hat = np.mean(bin_correctness)
            a_map[lo:hi] = a_hat
            u_hat = np.mean(uncertainty[lo:hi]) if hi > lo else None
            u_map[lo:hi] = u_hat
            a_hats.append(a_hat)
            u_hats.append(u_hat)
    sns.barplot(x=[round(u_hat, 2) for u_hat in u_hats], y=a_hats, ax=ax, **kwargs)
    ax.set_xlabel('Certainty')
    ax.set_ylabel('Correctness')
    ax.grid()
    fig.tight_layout()
    return ax

def histogram_alternative(correctness, uncertainties, fig, ax, num_bins=10, **kwargs):
    n = len(correctness)
    bin_endpoints = [round(ele) for ele in np.linspace(0, n, num_bins+1)]
    sorted_indices = np.argsort(uncertainties)
    correctness = correctness[sorted_indices]
    uncertainties = uncertainties[sorted_indices]
    a_map = -np.ones_like(correctness)
    u_map = np.zeros_like(uncertainties)
    a_hats = []
    u_hats = []
    # compute a_hat, u_hat and a_map: i -> a_hat, u_map: i -> u_hat
    for idx_bin in range(1, num_bins+1):
        lo, hi = bin_endpoints[idx_bin-1], bin_endpoints[idx_bin]
        if hi > lo:
            # if idx_bin == num_bins:  hi += 1
            bin_correctness = correctness[lo:hi]
            a_hat = np.mean(bin_correctness)
            a_map[lo:hi] = a_hat
            u_hat = np.mean(uncertainties[lo:hi]) if hi > lo else None
            u_map[lo:hi] = u_hat
            a_hats.append(a_hat)
            u_hats.append(u_hat)
    
    count_corrects = []
    count_uncertainties = []
    for a_hat, u_hat in zip(a_hats, u_hats):
        count_correct = (np.sum(a_map >= a_hat) - (1+np.sum(a_map == a_hat)) / 2) / (n-1)
        # count_uncertainty = (np.sum(u_map >= u_hat) - (1+np.sum(u_map == u_hat)) / 2) / (n-1)
        count_uncertainty = (np.sum(u_map <= u_hat) - (1+np.sum(u_map == u_hat)) / 2) / (n-1)
        # count_correct = (np.sum(a_map > a_hat) ) / (n-1)
        # count_uncertainty = (np.sum(u_map > u_hat) ) / (n-1)
        count_corrects.append(count_correct)
        count_uncertainties.append(count_uncertainty)
    sns.barplot(x=[round(count_uncertainty, 2) for count_uncertainty in count_uncertainties], y=count_corrects, ax=ax, **kwargs)
    ax.set_xlabel('Certainty')
    ax.set_ylabel('Correctness')
    ax.grid()
    fig.tight_layout()
    return ax