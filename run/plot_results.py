import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from utils import make_plots
epsilon = 1e-3

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='../tmp')
    # parser.add_argument('--file_names',  nargs='+', help='<Required> Set flag', required=True)
    args = parser.parse_args()

    # list all csv files in the root directory
    print(f"Loading files from {args.root_dir}")
    file_names = [file for file in os.listdir(args.root_dir) if file.endswith('.csv')]

    # Load the results from the previous task
    fig, ax = plt.subplots()
    for file_name in file_names:
        df = pd.read_csv(os.path.join(args.root_dir, file_name)).dropna(axis=0)
        correctness = df['score'].to_numpy()
        if 'semantic_entropy' in file_name:
            confidence = -df['confidence'].to_numpy()
        else:
            confidence = df['confidence'].to_numpy()
        thresholds = np.linspace(np.min(correctness)+epsilon, np.max(correctness)-epsilon, 10)
        ax = make_plots.AUROC_vs_Correctness(correctness, confidence, thresholds, ax=ax, label="_".join(file_name.split('_')[1:]))
    ax.set_title('AUROC vs Correctness Threshold')
    ax.figure.savefig('../tmp/auroc_vs_correctness.png')