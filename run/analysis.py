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
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--dataset', type=str, default='triviaqa')
    parser.add_argument('--affinity_mode', type=str, default='disagreement')
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
    elif args.correctness in ['chatgpt']:
        SCORE = correctness.ChatgptCorrectness()
    else:
        raise ValueError(f"Correctness metric {args.correctness} not found")

    model = args.model.split('/')[-1]
    collected_file = '_'.join([model, args.dataset, str(args.temperature)]) + '.json'
    collected = json.load(open(os.path.join('../collected', collected_file)))

    # compute the correctness score
    if os.path.exists(f'../tmp/{model}_{args.dataset}_{args.temperature}_{args.correctness}.json'):
        scores = json.load(open(f'../tmp/{model}_{args.dataset}_{args.temperature}_{args.correctness}.json'))
    else:
        from multiprocessing import Manager
        if args.correctness in ['rouge', 'bleu', 'meteor']:
            def compute_scores(col, base, scores):
                for idx, collected_row in tqdm(enumerate(col), total=len(col)):
                    score_tmp = {}
                    prompt = collected_row['prompt'].split('\n')[-2].strip()
                    reference = collected_row['references']
                    generations = collected_row['generated']
                    if args.correctness in ['rouge', 'bleu', 'meteor']:
                        s_unnormalized = [SCORE(references=[reference], predictions=[generation]) for generation in generations]
                        s_normalized = [SCORE(references=[reference], predictions=[generation]) for generation in generations]
                    else:
                        s_normalized = SCORE(prompt=prompt, references=reference, predictions=[generations])[0]
                        s_unnormalized = SCORE(prompt=prompt, references=reference, predictions=[generations])[0]
                    score_tmp['id'] = idx + base
                    score_tmp['normalized_score'] = s_normalized
                    score_tmp['unnormalized_score'] = s_unnormalized
                    scores.append(score_tmp)
            manager = Manager()
            scores = manager.list()
            processes = []
            def split(a, n):
                k, m = divmod(len(a), n)
                return [a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]
            collecteds = split(collected, 10)
            start = 0
            for col in collecteds:
                p = multiprocessing.Process(target=compute_scores, args=(col, start, scores))
                start += len(col)
                processes.append(p)
                p.start()
            for process in processes:
                process.join()
            scores = [score for score in scores]
        elif args.correctness in ['bert_similarity']:
            scores = []
            for idx, collected_row in tqdm(enumerate(collected), total=len(collected)):
                score_tmp = {}
                question = collected_row['prompt'].split('\n')[-2].strip()
                reference = collected_row['references'][0]
                generations = collected_row['generated']
                scores_tmp = SCORE(prompt=question, references=reference, predictions=generations)[0].tolist()
                score_tmp['id'] = idx
                score_tmp['normalized_score'] = scores_tmp
                score_tmp['unnormalized_score'] = scores_tmp
                scores.append(score_tmp)

                if idx % 10 == 0:
                    with open(f'../tmp/{model}_{args.dataset}_{args.temperature}_{args.correctness}.json', 'w') as f:
                        json.dump(scores, f)
        elif args.correctness in ['chatgpt']:
            scores = []
            for idx, collected_row in tqdm(enumerate(collected), total=len(collected)):
                if idx > 1000:
                    break
                score_tmp = {}
                question = collected_row['prompt'].split('\n')[-2].strip()
                reference = collected_row['references'][0]
                generations = collected_row['generated']
                scores_tmp = SCORE(prompt=question, reference=reference, generateds=generations)
                score_tmp['id'] = idx
                score_tmp['normalized_score'] = scores_tmp
                score_tmp['unnormalized_score'] = scores_tmp
                scores.append(score_tmp)

                if idx % 10 == 0:
                    with open(f'../tmp/{model}_{args.dataset}_{args.temperature}_{args.correctness}.json', 'w') as f:
                        json.dump(scores, f)
        with open(f'../tmp/{model}_{args.dataset}_{args.temperature}_{args.correctness}.json', 'w') as f:
            json.dump(scores, f)
        exit(0)
    scores = pd.DataFrame(scores).dropna(axis=0)
    
    if args.method == 'whitebox':
        affinity_mode = 'none'
    else:
        affinity_mode = args.affinity_mode
    file_name = "_".join(['calibrate', model, args.dataset, str(args.temperature), affinity_mode, args.method]) + '.json'
    file_name = os.path.join(args.root_dir, file_name)

    model = args.model.split('/')[-1]
    dataset = args.dataset
    method = args.method

    print('loading', os.path.join(args.root_dir, file_name))
    data = json.load(open(file_name))[:len(scores)]

    results = [] 
    for idx, row in tqdm(enumerate(data), total=len(data)):
        result = {'model':model, 'dataset':dataset, 'method':method, 
                    'metric':args.correctness, 'mode':args.mode}
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
        if method == 'whitebox':
            normalized_min_index = np.argmin(result['normalized_nll_all'])
            unnormalized_min_index = np.argmin(result['unnormalized_nll_all'])
            result['normalized_score_greedy'] = result['normalized_score_all'][normalized_min_index]
            result['unnormalized_score_greedy'] = result['unnormalized_score_all'][unnormalized_min_index]
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
        
        min_val = np.max(np.min(correctness_scores, axis=0))
        max_val = np.min(np.max(correctness_scores, axis=0))
        thresholds = np.linspace(min_val+epsilon, max_val-epsilon, 10)
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

    if method == 'whitebox':
        fig, ax = plt.subplots()
        confidence = -np.stack(df['unnormalized_nll_all']).flatten() if args.correctness != 'bert_similarity' else np.stack(df['unnormalized_nll_all']).flatten()
        ax = make_plots.histogram(correctness_scores, confidence, fig, ax)
        plt.savefig(f'{path}/erce_{model}_{dataset}_{affinity_mode}_unnormalized_nll_all_{args.correctness}.png')

        fig, ax = plt.subplots()
        threshold = 0.5
        y_true = correctness_scores >= threshold
        # plot roc curve
        fpr, tpr, _ = roc_curve(y_true, -confidence)
        ax.plot(fpr, tpr, label='ROC curve')
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC curve')
        ax.legend()
        ax.grid()
        plt.savefig(f'{path}/roc_curve_{model}_{dataset}_{affinity_mode}_{method}_{args.correctness}_entropy.png')
    else:
        fig, ax = plt.subplots()
        confidence = np.stack(df['degree_c']).flatten()
        ax = make_plots.histogram(correctness_scores, confidence, fig, ax)
        plt.savefig(f'{path}/erce_{model}_{dataset}_{affinity_mode}_ecc_u_{args.correctness}.png')

        fig, ax = plt.subplots()
        threshold = 0.5
        y_true = correctness_scores >= threshold
        # plot roc curve
        fpr, tpr, _ = roc_curve(y_true, confidence)
        ax.plot(fpr, tpr, label='ROC curve')
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC curve')
        ax.legend()
        ax.grid()
        plt.savefig(f'{path}/roc_curve_{model}_{dataset}_{affinity_mode}_{method}_{args.correctness}_degree.png')

    fig, ax = plt.subplots()
    ax.violinplot(df[indicators],
                showmeans=False,
                showmedians=True)
    ax.set_title('Uncertainty value distribution')
    ax.set_xticks([y+1 for y in range(len(indicators))], labels=indicators)
    plt.grid()
    plt.savefig(f'{path}/confidence_histogram_{model}_{dataset}_{affinity_mode}_{method}_{args.correctness}.png')