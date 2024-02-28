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
    if args.correctness in ['rouge', 'bleu', 'meteor']:
        SCORE = correctness.Score(metric_name=args.correctness, mode=args.mode)
    elif args.correctness in ['bert_similarity']:
        SCORE = correctness.BertSimilarity()
    
    for file_name in file_names:
        print(f'Loading {file_name}')
        model, dataset, affinity_mode, method = file_name.split('_')[1:5]
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
            if type(reference) != list:
                    reference = [reference] 
            if args.correctness in ['rouge', 'bleu', 'meteor']:
                s = SCORE(reference, generation_greedy)
            else:
                s = SCORE("", reference, generation_greedy)[0]
            result['score'] = s
            results.append(result)
        df = pd.DataFrame(results).dropna(axis=0)

        try: 
            path = os.path.join(args.root_dir, f"{model}_{dataset}_{affinity_mode}_blackbox_{args.correctness}")
            os.makedirs(path, exist_ok = True) 
        except OSError as error: 
            print("Directory '%s' can not be created" % path) 

        fig, ax = plt.subplots()
        correctness_score = df['score'].to_numpy()
        for indicator in indicators:
            confidence = -df[indicator].to_numpy()
            thresholds = np.linspace(np.min(correctness_score)+epsilon, np.max(correctness_score)-epsilon, 10)
            ax = make_plots.AUROC_vs_Correctness(correctness_score, confidence, thresholds, ax=ax, label=indicator)
        ax.set_title(f'AUROC vs Correctness Threshold {model} {dataset} {method} {args.correctness}')
        ax.grid()
        ax.figure.savefig(f'{path}/auroc_vs_correctness_{model}_{dataset}_{affinity_mode}_{method}_{args.correctness}.png')

        for indicator in indicators:
            fig, ax = plt.subplots()
            uncertainties = df[indicator].to_numpy()
            # plot a scatter plot of correctness score vs uncertainty
            ax.scatter(correctness_score, uncertainties)
            ax.set_title(f'Correctness score vs {indicator}')
            ax.set_xlabel('Correctness score')
            ax.set_ylabel(f'{indicator}')
            plt.grid()
            plt.savefig(f'{path}/erce_scatter_{model}_{dataset}_{affinity_mode}_{indicator}_{args.correctness}.png')

        fig, ax = plt.subplots()
        ax.violinplot(df[indicators],
                  showmeans=False,
                  showmedians=True)
        ax.set_title('Uncertainty value distribution')
        ax.set_xticks([y+1 for y in range(len(indicators))], labels=indicators)
        plt.grid()
        plt.savefig(f'{path}/confidence_histogram_{model}_{dataset}_{affinity_mode}_{method}_{args.correctness}.png')

        correctness_score = df['score'].to_numpy()
        # plot the histogram of correctness score
        fig, ax = plt.subplots()
        ax.hist(correctness_score, bins=20)
        ax.set_title('Correctness score distribution')
        ax.set_xlabel('Correctness score')
        ax.set_ylabel('Frequency')
        plt.grid()
        plt.savefig(f'{path}/correctness_histogram_{model}_{dataset}_{affinity_mode}_{method}_{args.correctness}.png')

        for indicator in indicators:
            fig, ax = plt.subplots()
            uncertainties = df[indicator].to_numpy()
            ax = make_plots.histogram(correctness_score, uncertainties, fig, ax)
            plt.savefig(f'{path}/erce_{model}_{dataset}_{affinity_mode}_{indicator}_{args.correctness}.png')

        for indicator in indicators:
            fig, ax = plt.subplots()
            uncertainties = df[indicator].to_numpy()
            ax = make_plots.histogram_alternative(correctness_score, uncertainties, fig, ax)
            plt.savefig(f'{path}/erce_alternative{model}_{dataset}_{affinity_mode}_{indicator}_{args.correctness}.png')

        fig, ax = plt.subplots()
        ax.violinplot(df[indicators],
                  showmeans=False,
                  showmedians=True)
        ax.set_title('Uncertainty value distribution')
        ax.set_xticks([y+1 for y in range(len(indicators))], labels=indicators)
        plt.grid()
        plt.savefig(f'{path}/confidence_histogram_{model}_{dataset}_{affinity_mode}_{method}_{args.correctness}.png')
        
        if 'entropy' in indicators:
            fig, ax = plt.subplots()
            threshold = 0.5
            y_true = correctness_score >= threshold
            y_score = -df['entropy']
            # plot roc curve
            fpr, tpr, _ = roc_curve(y_true, y_score)
            ax.plot(fpr, tpr, label='ROC curve')
            ax.plot([0, 1], [0, 1], 'k--', label='Random')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC curve')
            ax.legend()
            ax.grid()
            plt.savefig(f'{path}/roc_curve_{model}_{dataset}_{affinity_mode}_{method}_{args.correctness}_entropy.png')
        
        if 'degree' in indicators:
            fig, ax = plt.subplots()
            threshold = 0.5
            y_true = correctness_score >= threshold
            y_score = -df['degree']
            # plot roc curve
            fpr, tpr, _ = roc_curve(y_true, y_score)
            ax.plot(fpr, tpr, label='ROC curve')
            ax.plot([0, 1], [0, 1], 'k--', label='Random')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC curve')
            ax.legend()
            ax.grid()
            plt.savefig(f'{path}/roc_curve_{model}_{dataset}_{affinity_mode}_{method}_{args.correctness}_degree.png')
