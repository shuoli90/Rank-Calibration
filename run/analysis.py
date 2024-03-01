import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
import multiprocessing
import json
from sklearn.metrics import roc_curve
from utils import make_plots
from metrics import correctness
epsilon = 1e-3

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='../tmp')
    parser.add_argument('--correctness', type=str, default='rouge')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--dataset', type=str, default='triviaqa')
    parser.add_argument('--affinity_mode', type=str, default='none')
    parser.add_argument('--method', type=str, default='whitebox')
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

    model = args.model.split('/')[-1]
    # compute the correctness score
    if os.path.exists(f'../tmp/{model}_{args.dataset}_{args.correctness}.json'):
        scores = json.load(open(f'../tmp/{model}_{args.dataset}_{args.correctness}.json'))
    else:
        from multiprocessing import Manager
        
        collected_file = '_'.join([model, args.dataset]) + '.json'
        collected = json.load(open(os.path.join('../collected', collected_file)))

        def compute_scores(col, base, scores):
            for idx, collected_row in tqdm(enumerate(col), total=len(col)):
                score_tmp = {}
                reference = collected_row['references']
                generations = collected_row['generated']
                if args.correctness in ['rouge', 'bleu', 'meteor']:
                    s_unnormalized = [SCORE(references=[reference], predictions=[generation]) for generation in generations]
                    s_normalized = [SCORE(references=[reference], predictions=[generation]) for generation in generations]
                else:
                    s_normalized = SCORE(references=reference, predictions=[generations])[0]
                    s_unnormalized = SCORE(references=reference, predictions=[generations])[0]
                score_tmp['id'] = idx + base
                score_tmp['normalized_score'] = s_normalized
                score_tmp['unnormalized_score'] = s_unnormalized
                scores.append(score_tmp)

        manager = Manager()
        scores = manager.list()
        processes = []
        # divide the collected into K chunks
        K = 10
        chunk_size = len(collected) // K
        collecteds = [collected[i*chunk_size:(i+1)*chunk_size] for i in range(K)]
        for i, col in enumerate(collecteds):
            p = multiprocessing.Process(target=compute_scores, args=(col, i*chunk_size, scores))
            processes.append(p)
            p.start()
        for process in processes:
            process.join()
        scores = [score for score in scores]
        with open(f'../tmp/{model}_{args.dataset}_{args.correctness}.json', 'w') as f:
            json.dump(scores, f)
    scores = pd.DataFrame(scores).dropna(axis=0)
    breakpoint()
    
    file_name = "_".join(['calibrate', args.model, args.dataset, args.affinity_mode, args.method]) + '.json'
    file_name = os.path.join(args.root_dir, file_name)

    model = args.model.split('/')[-1]
    dataset = args.dataset
    affinity_mode = args.affinity_mode
    method = args.method

    print('loading', os.path.join(args.root_dir, file_name))
    data = json.load(open(os.path.join(args.root_dir, file_name)))

    results = [] 
    for idx, (collected_row, row) in tqdm(enumerate(zip(collected, data)), total=len(data)):
        if idx > 10:
            break
        result = {'model':model, 'dataset':dataset, 'method':method, 
                    'metric':args.correctness, 'mode':args.mode}
        reference = collected_row['references']
        generations = collected_row['generated']
        if method == 'blackbox':
            result['ecc_u'] = row['ecc_u']
            result['ecc_c'] = row['ecc_c']
            result['degree_u'] = row['degree_u']
            result['degree_c'] = row['degree_c']
            result['spectral_u'] = row['spectral_u']
        elif method == 'whitebox':
            normalized_nll = row['normalized_nll']
            unnormalized_nll = row['unnormalized_nll']
            entropy_normalized = row['entropy_normalized']
            entropy_unnormalized = row['entropy_unnormalized']
            result['normalized_nll_all'] = normalized_nll
            result['unnormalized_nll_all'] = unnormalized_nll
            result['normalized_nll_greedy'] = np.min(normalized_nll)
            result['unnormalized_nll_greedy'] = np.min(unnormalized_nll)
            result['entropy_normalized'] = entropy_normalized
            result['entropy_unnormalized'] = entropy_unnormalized  
        # select scores with the same index
        score = scores[scores['id'] == idx]
        result['normalized_score_all'] = score.iloc[0]['normalized_score']
        result['unnormalized_score_all'] = score.iloc[0]['unnormalized_score']
        results.append(result)
    df = pd.DataFrame(results).dropna(axis=0)
    scores = pd.DataFrame(scores).dropna(axis=0)
    # concatenate the scores
    df = pd.concat([df, scores], axis=1)

    try: 
        path = os.path.join(args.root_dir, f"{model}_{dataset}_{affinity_mode}_{method}_{args.correctness}")
        os.makedirs(path, exist_ok = True) 
    except OSError as error: 
        print("Directory '%s' can not be created" % path) 

    if method == 'whitebox':
        fig, ax = plt.subplots()
        # correctness_score = df['score'].to_numpy()
        correctness_score = df['normalized_score_greedy'].to_numpy()
        for indicator in ['normalized_nll_greedy', 'unnormalized_nll_greedy', 'entropy_normalized', 'entropy_unnormalized']:
            confidence = -df[indicator].to_numpy()
            thresholds = np.linspace(np.min(correctness_score)+epsilon, np.max(correctness_score)-epsilon, 10)
            ax = make_plots.AUROC_vs_Correctness(correctness_score, confidence, thresholds, ax=ax, label=indicator)
        ax.set_title(f'AUROC vs Correctness Threshold {model} {dataset} {method} {args.correctness}')
        ax.grid()
        ax.figure.savefig(f'{path}/auroc_vs_correctness_{model}_{dataset}_{affinity_mode}_{method}_{args.correctness}.png')

    if method == 'whitebox':
        indicators = ['normalized_nll_all', 'unnormalized_nll_all']
    else:
        indicators = ['ecc_c', 'degree_c']

    correctness_scores = np.stack(df['normalized_score_all'])
    fig, ax = plt.subplots()
    for indicator in indicators:
        if method == 'whitebox':
            confidence = -np.stack(df[indicator])
        else:
            confidence = np.stack(df[indicator])
        thresholds = np.linspace(0.0+epsilon, 1.0-epsilon, 10)
        ax = make_plots.AUROC_vs_Correctness_average(correctness_scores, confidence, thresholds, ax=ax, label=indicator)
    ax.set_title(f'AUROC vs Correctness Threshold {model} {dataset} {method} {args.correctness}')
    ax.grid()
    ax.figure.savefig(f'{path}/auroc_vs_correctness_average_{model}_{dataset}_{affinity_mode}_{method}_{args.correctness}.png')

    # correctness_score = df['normalized_score_greedy'].to_numpy()
    correctness_scores = np.stack(df['normalized_score_all']).flatten()
    for indicator in indicators:
        fig, ax = plt.subplots()
        if method == 'whitebox':
            confidence = -np.stack(df[indicator]).flatten()
        else:
            confidence = np.stack(df[indicator]).flatten()
        # plot a scatter plot of correctness score vs uncertainty
        ax.scatter(correctness_scores, confidence)
        ax.set_title(f'Correctness score vs {indicator}')
        ax.set_xlabel('Correctness score')
        ax.set_ylabel(f'{indicator}')
        plt.grid()
        plt.savefig(f'{path}/erce_scatter_{model}_{dataset}_{affinity_mode}_{indicator}_{args.correctness}.png')

    if method == 'whitebox':
        indicators = ['normalized_nll_greedy', 'unnormalized_nll_greedy', 'entropy_normalized', 'entropy_unnormalized']
    else:
        indicators = ['ecc_u', 'degree_u', 'spectral_u']
    fig, ax = plt.subplots()
    ax.violinplot(df[indicators],
                showmeans=False,
                showmedians=True)
    ax.set_title('Uncertainty value distribution')
    ax.set_xticks([y+1 for y in range(len(indicators))], labels=indicators)
    plt.grid()
    plt.savefig(f'{path}/confidence_histogram_{model}_{dataset}_{affinity_mode}_{method}_{args.correctness}.png')

    # plot the histogram of correctness score
    fig, ax = plt.subplots()
    ax.hist(correctness_scores, bins=20)
    ax.set_title('Correctness score distribution')
    ax.set_xlabel('Correctness score')
    ax.set_ylabel('Frequency')
    plt.grid()
    plt.savefig(f'{path}/correctness_histogram_{model}_{dataset}_{affinity_mode}_{method}_{args.correctness}.png')

    for indicator in indicators:
        fig, ax = plt.subplots()
        if method == 'whitebox':
            confidence = -np.stack(df[indicator]).flatten()
        else:
            confidence = np.stack(df[indicator]).flatten()
        ax = make_plots.histogram(correctness_scores, confidence, fig, ax)
        plt.savefig(f'{path}/erce_{model}_{dataset}_{affinity_mode}_{indicator}_{args.correctness}.png')

    for indicator in indicators:
        fig, ax = plt.subplots()
        if method == 'whitebox':
            confidence = -np.stack(df[indicator]).flatten()
        else:
            confidence = np.stack(df[indicator]).flatten()
        ax = make_plots.histogram_alternative(correctness_scores, confidence, fig, ax)
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