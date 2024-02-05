import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import pandas as pd
import random
from metrics import ranking

if __name__ == '__main__':
    # indicators = ['jaccard', 'semantic_consistency', 'spectral_projected', 'eccentricity', 'degree']
    # metrics = ['auarc', 'auroc', 'f1']
    # # compute all combinations of indicators and metrics
    # combinations = [(indicator, metric) for indicator in indicators for metric in metrics]
    # # randomly generate scores for each combination
    # scores = [random.random() for _ in range(len(combinations))]
    # data = {'indicator': [indicator for indicator, _ in combinations],  
    #         'metric': [metric for _, metric in combinations], 
    #         'score': scores}
    # df = pd.DataFrame(data)
    df = pd.read_csv('example.csv')
    ranking.plot_cd_diagram(df, title='test', save_dir='../tmp/test.png', col1='indicators', col2='metrics', col3='scores')