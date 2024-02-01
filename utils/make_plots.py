import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_auc_score

def AUROC_vs_Correctness(correctness, confidence, thresholds, ax, **kwargs):
    # compute AUROC for different correctness thresholds
    aurocs = []
    for threshold in thresholds:
        y_true = correctness >= threshold
        y_score = confidence
        auroc = roc_auc_score(y_true, y_score)
        # auroc = max(auroc, 1-auroc)
        aurocs.append(auroc)
    # plot
    df = pd.DataFrame(dict(AUROC=aurocs, Correctness=thresholds))
    sns.lineplot(x="Correctness", y="AUROC", data=df, ax=ax, **kwargs)
    return ax