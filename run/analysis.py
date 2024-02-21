import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from sklearn.metrics import roc_curve
from utils import make_plots
from metrics import correctness
epsilon = 1e-3

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='../tmp')
    parser.add_argument('--correctness', type=str, default='rouge')
    parser.add_argument('--mode', type=str, default='rougeL')

    args = parser.parse_args()

    # list all csv files in the root directory
    print(f"Loading files from {args.root_dir}")
    file_names = [file for file in os.listdir(args.root_dir) if file.endswith('.json')]

    # setup generation correctness score
    SCORE = correctness.Score(metric_name=args.correctness, mode=args.mode)

    for file_name in file_names:
        if "blackbox" not in file_name:
            continue
        print(f'Loading {file_name}')
        model, dataset, method = file_name.split('_')[1:]
        method = method.split('.')[0]
        print('loading', os.path.join(args.root_dir, file_name))
        data = json.load(open(os.path.join(args.root_dir, file_name)))
        print('loading', os.path.join('../collected', f'{model}_{dataset}.json'))
        collected = json.load(open(os.path.join('../collected', f'{model}_{dataset}.json')))[:len(data)]
        results = [] 
        for collected_row, row in zip(collected, data):
            result = {'model':model, 'dataset':dataset, 'method':method, 
                      'metric':args.correctness, 'mode':args.mode}
            reference = collected_row['references'][0]
            generation_greedy = collected_row['greedy']
            generation_sampled = collected_row['sampled']
            if method == 'blackbox':
                indicators = ['ecc', 'degree', 'spectral']
                ecc = row['ecc_u']
                degree = row['degree_u']
                spectral = row['spectral_u']
                result['ecc'] = ecc
                result['degree'] = degree
                result['spectral'] = spectral
            elif method == 'whitebox':
                indicators = ['normalized_nll', 'unnormalized_nll', 'entropy']
                normalized_nll = row['normalized_nll']
                unnormalized_nll = row['unnormalized_nll']
                semantic_entropy = row['entropy']
                result['normalized_nll'] = normalized_nll
                result['unnormalized_nll'] = unnormalized_nll
                result['entropy'] = semantic_entropy
            s = SCORE(reference, generation_greedy)
            result['score'] = s
            results.append(result)
        df = pd.DataFrame(results).dropna(axis=0)

        # fig, ax = plt.subplots()
        # correctness = df['score'].to_numpy()
        # for indicator in indicators:
        #     confidence = -df[indicator].to_numpy()
        #     thresholds = np.linspace(np.min(correctness)+epsilon, np.max(correctness)-epsilon, 10)
        #     ax = make_plots.AUROC_vs_Correctness(correctness, confidence, thresholds, ax=ax, label=indicator)
        # ax.set_title(f'AUROC vs Correctness Threshold {model} {dataset} {method}')
        # ax.grid()
        # ax.figure.savefig(f'../tmp/auroc_vs_correctness_{model}_{dataset}_{method}.png')

        # fig, ax = plt.subplots()
        # ax.violinplot(df[indicators],
        #           showmeans=False,
        #           showmedians=True)
        # ax.set_title('Uncertainty value distribution')
        # ax.set_xticks([y+1 for y in range(len(indicators))], labels=indicators)
        # plt.grid()
        # plt.savefig(f'../tmp/confidence_histogram_{model}_{dataset}_{method}.png')
        correctness_score = df['score'].to_numpy()
        for indicator in indicators:
            fig, ax = plt.subplots()
            uncertainties = df[indicator].to_numpy()
            ax = make_plots.histogram(correctness_score, uncertainties, fig, ax)
            plt.savefig(f'../tmp/erce_{model}_{dataset}_{indicator}.png')
            breakpoint()

        # if 'degree' in indicators:
        #     fig, ax = plt.subplots()
        #     threshold = 0.5
        #     y_true = correctness >= threshold
        #     y_score = -df['degree']
        #     # plot roc curve
        #     fpr, tpr, _ = roc_curve(y_true, y_score)
        #     ax.plot(fpr, tpr, label='ROC curve')
        #     ax.plot([0, 1], [0, 1], 'k--', label='Random')
        #     ax.set_xlabel('False Positive Rate')
        #     ax.set_ylabel('True Positive Rate')
        #     ax.set_title('ROC curve')
        #     ax.legend()
        #     ax.grid()
        #     plt.savefig(f'../tmp/roc_curve_{model}_{dataset}_{method}_degree.png')
        