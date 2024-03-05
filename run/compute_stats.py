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
    parser.add_argument('--mode', type=str, default='rougeL')
    args = parser.parse_args()

    # list all csv files in the root directory
    print(f"Loading files from {args.root_dir}")
    file_names = [file for file in os.listdir(args.root_dir) if file.endswith('.json')]
    model = args.model.split('/')[-1]
    # compute the correctness score
    if os.path.exists(f'../tmp/{model}_{args.dataset}_{args.temperature}_{args.correctness}.json'):
        scores = json.load(open(f'../tmp/{model}_{args.dataset}_{args.temperature}_{args.correctness}.json'))
    else:
        model = args.model.split('/')[-1]
        collected_file = '_'.join([model, args.dataset, str(args.temperature)]) + '.json'
        collected = json.load(open(os.path.join('../collected', collected_file)))
         # setup generation correctness score
        if args.correctness in ['rouge', 'bleu', 'meteor']:
            SCORE = correctness.Score(metric_name=args.correctness, mode=args.mode)
        elif args.correctness in ['bert_similarity']:
            SCORE = correctness.BertSimilarity()
        elif args.correctness in ['chatgpt']:
            SCORE = correctness.ChatgptCorrectness()
        else:
            raise ValueError(f"Correctness metric {args.correctness} not found")
    
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

    model = args.model.split('/')[-1]
    dataset = args.dataset
    file_names = []
    for method in ['whitebox', 'blackbox']:
        if method == 'whitebox':
            affinity_mode = 'none'
            file_name = "_".join(['calibrate', model, dataset, str(args.temperature), affinity_mode, 'whitebox']) + '.json'
            file_names.append(file_name)
        else:
            for affinity_mode in ['disagreement', 'agreement']:
                file_name = "_".join(['calibrate', model, dataset, str(args.temperature), affinity_mode, 'blackbox']) + '.json'
                file_names.append(file_name)
    data_whitebox = json.load(open(os.path.join(args.root_dir, file_names[0])))
    data_blackbox_disagreement = json.load(open(os.path.join(args.root_dir, file_names[1])))
    data_blackbox_agreement = json.load(open(os.path.join(args.root_dir, file_names[2])))
    results = []
    for idx, (row_whitebox, row_blackbox_disagreement, row_blackbox_agreement) in tqdm(enumerate(zip(data_whitebox, data_blackbox_disagreement, data_blackbox_agreement)), total=len(data_whitebox)):
        result = {'model':model, 'dataset':dataset, 'method':method, 
                    'metric':args.correctness, 'mode':args.mode}
        
        result['ecc_c_disagreement'] = row_blackbox_disagreement['ecc_c']
        result['degree_c_disagreement'] = row_blackbox_disagreement['degree_c']
        result['ecc_u_disagreement'] = row_blackbox_disagreement['ecc_u']
        result['degree_u_disagreement'] = row_blackbox_disagreement['degree_u']
        result['spectral_u_disagreement'] = row_blackbox_disagreement['spectral_u']

        result['ecc_c_agreement'] = row_blackbox_agreement['ecc_c']
        result['degree_c_agreement'] = row_blackbox_agreement['degree_c']
        result['ecc_u_agreement'] = row_blackbox_agreement['ecc_u']
        result['degree_u_agreement'] = row_blackbox_agreement['degree_u']
        result['spectral_u_agreement'] = row_blackbox_agreement['spectral_u']

        result['entropy_normalized'] = row_whitebox['entropy_normalized']
        result['entropy_unnormalized'] = row_whitebox['entropy_unnormalized']
        result['normalized_nll_all'] = row_whitebox['normalized_nll']
        result['unnormalized_nll_all'] = row_whitebox['unnormalized_nll']
        result['normalized_nll_greedy'] = np.min(row_whitebox['normalized_nll'])
        result['unnormalized_nll_greedy'] = np.min(row_whitebox['unnormalized_nll'])

        # select scores with the same index
        score = scores[scores['id'] == idx]
        result['normalized_score_all'] = score.iloc[0]['normalized_score']
        result['unnormalized_score_all'] = score.iloc[0]['unnormalized_score']
        normalized_min_index = np.argmin(result['normalized_nll_all'])
        unnormalized_min_index = np.argmin(result['unnormalized_nll_all'])
        result['normalized_score_greedy'] = result['normalized_score_all'][normalized_min_index]
        result['unnormalized_score_greedy'] = result['unnormalized_score_all'][unnormalized_min_index]
        results.append(result)
    df = pd.DataFrame(results).dropna(axis=0)
    scores = pd.DataFrame(scores).dropna(axis=0)
    # concatenate the scores
    df = pd.concat([df, scores], axis=1)

    uncertainty_indicators = ['ecc_u_disagreement', 'degree_u_disagreement', 'spectral_u_disagreement',
                              'ecc_u_agreement', 'degree_u_agreement', 'spectral_u_agreement',
                              'normalized_nll_greedy', 'unnormalized_nll_greedy', 'entropy_normalized', 'entropy_unnormalized']
    confidence_indicators = ['ecc_c_disagreement', 'degree_c_disagreement', 'ecc_c_agreement', 'degree_c_agreement', 
                             'normalized_nll_all', 'unnormalized_nll_all']
    indicators = uncertainty_indicators + confidence_indicators

    # uncertainty aurocs 
    uncertainty_aurocs = {}
    correctness_score = df['normalized_score_greedy'].to_numpy()
    for indicator in uncertainty_indicators:
        confidence = -df[indicator].to_numpy()
        thresholds = np.linspace(np.min(correctness_score)+epsilon, np.max(correctness_score)-epsilon, 10)
        aurocs = make_plots.AUROC_vs_Correctness(correctness_score, confidence, thresholds, ax=None, plot=False, label=indicator)
        uncertainty_aurocs[indicator] = aurocs

    confidence_aurocs = {} 
    correctness_scores = np.stack(df['normalized_score_all'])
    for indicator in confidence_indicators:
        confidence = np.stack(df[indicator]) if 'agreement' in indicator else -np.stack(df[indicator])
        min_val = np.max(np.min(correctness_scores, axis=0))
        max_val = np.min(np.max(correctness_scores, axis=0))
        thresholds = np.linspace(min_val+epsilon, max_val-epsilon, 10)
        aurocs = make_plots.AUROC_vs_Correctness_average(correctness_scores, confidence, thresholds, ax=None, plot=False, label=indicator)
        uncertainty_aurocs[indicator] = aurocs  
    
    uncertainty_erces = {}
    correctness_scores = np.stack(df['normalized_score_greedy']).flatten()
    for indicator in uncertainty_indicators:
        uncertainty = df[indicator].to_numpy()
        erce = make_plots.histogram_alternative(correctness=correctness_scores, uncertainties=uncertainty, ax=None, plot=None, num_bins=50)
        uncertainty_erces[indicator] = erce
    
    confidence_erces = {}
    correctness_scores = np.stack(df['normalized_score_all']).flatten()
    for indicator in confidence_indicators:
        confidence = -np.stack(df[indicator]) if 'agreement' in indicator else np.stack(df[indicator])
        erce = make_plots.histogram_alternative(correctness=correctness_scores, uncertainties=confidence, ax=None, plot=False, num_bins=50)
        confidence_erces[indicator] = erce