import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import multiprocessing
import json
from sklearn.metrics import roc_curve
from utils import make_plots
from metrics import correctness, calibration
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
        raise ValueError(f"File not found: {model}_{args.dataset}_{args.temperature}_{args.correctness}.json")
    scores = pd.DataFrame(scores).dropna(axis=0)

    model = args.model.split('/')[-1]
    dataset = args.dataset
    file_names = []
    for method in ['whitebox', 'blackbox', 'verbalized']:
        if method == 'whitebox':
            affinity_mode = 'none'
            file_name = "_".join(['calibrate', model, dataset, str(args.temperature), affinity_mode, 'whitebox']) + '.json'
            file_names.append(file_name)
        elif method == 'verbalized':
            try:
                file_name = "_".join(['calibrate', model, dataset, str(args.temperature), 'disagreement', 'verbalized']) + '.json'
                file_names.append(file_name)
            except:
                continue
        else:
            for affinity_mode in ['disagreement', 'agreement']:
                file_name = "_".join(['calibrate', model, dataset, str(args.temperature), affinity_mode, 'blackbox']) + '.json'
                file_names.append(file_name)
    data_whitebox = json.load(open(os.path.join(args.root_dir, file_names[0])))
    data_blackbox_disagreement = json.load(open(os.path.join(args.root_dir, file_names[1])))
    data_blackbox_agreement = json.load(open(os.path.join(args.root_dir, file_names[2])))

    results = []
    seeds = list(range(20))
    for seed in seeds:
        np.random.seed(seed)
        if model == 'gpt-3.5-turbo' and args.temperature == 1.0:
            data_verbalized = json.load(open(os.path.join(args.root_dir, file_names[3])))
            indices = np.random.choice(len(data_verbalized), len(data_verbalized), replace=True).tolist()
            data_verbalized_bootstrap = [data_verbalized[index] for index in indices]
            tmps = []
            for row_verbalized in tqdm(data_verbalized_bootstrap):
                tmp = {'model':model, 'dataset':dataset, 'metric':args.correctness, 'seed':seed, 'temperature':args.temperature}
                idx = row_verbalized['idx']
                row_whitebox = data_whitebox[idx]
                row_blackbox_disagreement = data_blackbox_disagreement[idx]
                row_blackbox_agreement = data_blackbox_agreement[idx]
                
                tmp['ecc_c_disagreement'] = row_blackbox_disagreement['ecc_c']
                tmp['degree_c_disagreement'] = row_blackbox_disagreement['degree_c']
                tmp['ecc_u_disagreement'] = [row_blackbox_disagreement['ecc_u']] * 10
                tmp['degree_u_disagreement'] = [row_blackbox_disagreement['degree_u']] * 10
                tmp['spectral_u_disagreement'] = [row_blackbox_disagreement['spectral_u']] * 10
                tmp['verbalized'] = row_verbalized['verbalized']

                tmp['ecc_c_agreement'] = row_blackbox_agreement['ecc_c']
                tmp['degree_c_agreement'] = row_blackbox_agreement['degree_c']
                tmp['ecc_u_agreement'] = [row_blackbox_agreement['ecc_u']] * 10
                tmp['degree_u_agreement'] = [row_blackbox_agreement['degree_u']] * 10
                tmp['spectral_u_agreement'] = [row_blackbox_agreement['spectral_u']] * 10

                tmp['entropy_normalized'] = [row_whitebox['entropy_normalized']] * 10
                tmp['entropy_unnormalized'] = [row_whitebox['entropy_unnormalized']] * 10
                tmp['normalized_nll_all'] = row_whitebox['normalized_nll']
                tmp['unnormalized_nll_all'] = row_whitebox['unnormalized_nll']

                # select scores with the same index
                score = scores[scores['id'] == idx]
                tmp['normalized_score_all'] = score.iloc[0]['normalized_score']
                tmp['unnormalized_score_all'] = score.iloc[0]['unnormalized_score']
                normalized_min_index = np.argmin(tmp['normalized_nll_all'])
                unnormalized_min_index = np.argmin(tmp['unnormalized_nll_all'])
                tmp['normalized_score_greedy'] = tmp['normalized_score_all'][normalized_min_index]
                tmp['unnormalized_score_greedy'] = tmp['unnormalized_score_all'][unnormalized_min_index]
                tmps.append(tmp)
        else:
            # sample with replacement from the indices of the data
            indices = np.random.choice(len(data_whitebox), len(data_whitebox), replace=True).tolist()
            data_whitebox_bootstrap = [data_whitebox[index] for index in indices]
            data_blackbox_disagreement_bootstrap = [data_blackbox_disagreement[index] for index in indices]
            data_blackbox_agreement_bootstrap = [data_blackbox_agreement[index] for index in indices]
            tmps = []
            for idx, (index, row_whitebox, row_blackbox_disagreement, row_blackbox_agreement) in tqdm(enumerate(zip(indices, data_whitebox_bootstrap, data_blackbox_disagreement_bootstrap, data_blackbox_agreement_bootstrap)), total=len(data_whitebox)):
                tmp = {'model':model, 'dataset':dataset, 'metric':args.correctness, 'seed':seed, 'temperature':args.temperature}

                tmp['ecc_c_disagreement'] = row_blackbox_disagreement['ecc_c']
                tmp['degree_c_disagreement'] = row_blackbox_disagreement['degree_c']
                tmp['ecc_u_disagreement'] = [row_blackbox_disagreement['ecc_u']] * 10
                tmp['degree_u_disagreement'] = [row_blackbox_disagreement['degree_u']] * 10
                tmp['spectral_u_disagreement'] = [row_blackbox_disagreement['spectral_u']] * 10

                tmp['ecc_c_agreement'] = row_blackbox_agreement['ecc_c']
                tmp['degree_c_agreement'] = row_blackbox_agreement['degree_c']
                tmp['ecc_u_agreement'] = [row_blackbox_agreement['ecc_u']] * 10
                tmp['degree_u_agreement'] = [row_blackbox_agreement['degree_u']] * 10
                tmp['spectral_u_agreement'] = [row_blackbox_agreement['spectral_u']] * 10

                tmp['entropy_normalized'] = [row_whitebox['entropy_normalized']] * 10
                tmp['entropy_unnormalized'] = [row_whitebox['entropy_unnormalized']] * 10
                tmp['normalized_nll_all'] = row_whitebox['normalized_nll']
                tmp['unnormalized_nll_all'] = row_whitebox['unnormalized_nll']

                # select scores with the same index
                score = scores[scores['id'] == index]
                tmp['normalized_score_all'] = score.iloc[0]['normalized_score']
                tmp['unnormalized_score_all'] = score.iloc[0]['unnormalized_score']
                normalized_min_index = np.argmin(tmp['normalized_nll_all'])
                unnormalized_min_index = np.argmin(tmp['unnormalized_nll_all'])
                tmp['normalized_score_greedy'] = tmp['normalized_score_all'][normalized_min_index]
                tmp['unnormalized_score_greedy'] = tmp['unnormalized_score_all'][unnormalized_min_index]
                tmps.append(tmp)
        df = pd.DataFrame(tmps).dropna(axis=0)

        if model == 'gpt-3.5-turbo' and args.temperature == 1.0:
            uncertainty_indicators = ['ecc_u_agreement', 'degree_u_agreement', 'spectral_u_agreement', 'verbalized',
                                    'normalized_nll_all', 'unnormalized_nll_all', 'entropy_normalized', 'entropy_unnormalized']
        else:
            uncertainty_indicators = ['ecc_u_agreement', 'degree_u_agreement', 'spectral_u_agreement', 
                                    'normalized_nll_all', 'unnormalized_nll_all', 'entropy_normalized', 'entropy_unnormalized']
        
        correctness_scores = np.stack(df['normalized_score_all']).flatten()
        for indicator in uncertainty_indicators:
            uncertainty = np.stack(df[indicator]).flatten() if 'verbalized' not in indicator else -np.stack(df[indicator]).flatten()
            erce = calibration.plugin_RCE_est(correctness=correctness_scores, uncertainties=uncertainty, num_bins=20, p=1)
            tmp[f'{indicator}_erce'] = erce
    
        results.append(tmp)
    
    # write the results to a file
    with open(f'../stats/{model}_{dataset}_{args.temperature}_{args.correctness}.json', 'w') as f:
        json.dump(results, f)
    