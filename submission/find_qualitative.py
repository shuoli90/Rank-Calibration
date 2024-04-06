import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import json
epsilon = 1e-3

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='../tmp')
    parser.add_argument('--correctness', type=str, default='rouge')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--dataset', type=str, default='triviaqa')
    parser.add_argument('--mode', type=str, default='rougeL')
    args = parser.parse_args()

    # read in collected data
    model = args.model.split('/')[-1]
    collected_file = '_'.join([model, args.dataset, str(args.temperature)]) + '.json'
    collected = json.load(open(os.path.join('../collected', collected_file)))

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
    indices = np.arange(len(data_whitebox))

    results = []
    for idx, row_whitebox, row_blackbox_disagreement, row_blackbox_agreement in tqdm(zip(indices, data_whitebox, data_blackbox_disagreement, data_blackbox_agreement), total=len(data_whitebox)):
        tmp = {}
        tmp['ecc_c_disagreement'] = row_blackbox_disagreement['ecc_c']
        tmp['degree_c_disagreement'] = row_blackbox_disagreement['degree_c']
        tmp['ecc_u_disagreement'] = row_blackbox_disagreement['ecc_u']
        tmp['degree_u_disagreement'] = row_blackbox_disagreement['degree_u']
        tmp['spectral_u_disagreement'] = row_blackbox_disagreement['spectral_u']

        tmp['ecc_c_agreement'] = row_blackbox_agreement['ecc_c']
        tmp['degree_c_agreement'] = row_blackbox_agreement['degree_c']
        tmp['ecc_u_agreement'] = row_blackbox_agreement['ecc_u']
        tmp['degree_u_agreement'] = row_blackbox_agreement['degree_u']
        tmp['spectral_u_agreement'] = row_blackbox_agreement['spectral_u']

        tmp['entropy_normalized'] = row_whitebox['entropy_normalized']
        tmp['entropy_unnormalized'] = row_whitebox['entropy_unnormalized']
        tmp['normalized_nll_all'] = row_whitebox['normalized_nll']
        tmp['unnormalized_nll_all'] = row_whitebox['unnormalized_nll']
        tmp['normalized_nll_greedy'] = np.min(row_whitebox['normalized_nll'])
        tmp['unnormalized_nll_greedy'] = np.min(row_whitebox['unnormalized_nll'])

        # select scores with the same index
        score = scores[scores['id'] == idx]
        tmp['normalized_score_all'] = score.iloc[0]['normalized_score']
        tmp['unnormalized_score_all'] = score.iloc[0]['unnormalized_score']
        normalized_min_index = np.argmin(tmp['normalized_nll_all'])
        unnormalized_min_index = np.argmin(tmp['unnormalized_nll_all'])
        tmp['greedy_index'] = int(normalized_min_index)
        tmp['normalized_score_greedy'] = tmp['normalized_score_all'][normalized_min_index]
        tmp['unnormalized_score_greedy'] = tmp['unnormalized_score_all'][unnormalized_min_index]
        results.append(tmp)

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

    def cdf(arr):
        result = []
        for i in range(len(arr)):
            result.append(np.sum(arr <= arr[i])/len(arr))
        return np.array(result)
    
    df_tmp = df[uncertainty_indicators+['normalized_score_greedy', 'greedy_index']]
    df_tmp[uncertainty_indicators+['normalized_score_greedy']] = df_tmp[uncertainty_indicators+['normalized_score_greedy']].apply(lambda x: cdf(x))

    # iterate through 
    quals = []
    for idx, row in df_tmp.iterrows():
        greedy_index = int(row['greedy_index'])
        prompt = collected[idx]['prompt'].split('\n')[-2].strip()
        references = collected[idx]['references']
        generated = collected[idx]['generated'][greedy_index]
        ecc = 1-row['ecc_u_agreement']
        degree = 1-row['degree_u_agreement']
        spectral = 1-row['spectral_u_agreement']
        nll = 1-row['unnormalized_nll_greedy']
        entropy = 1-row['entropy_unnormalized']
        score = row['normalized_score_greedy']
        residual_ecc = np.abs(ecc - score)
        residual_degree = np.abs(degree - score)
        residual_spectral = np.abs(spectral - score)
        residual_nll = np.abs(nll - score)
        residual_entropy = np.abs(entropy - score)
        residuals = [residual_ecc, residual_degree, residual_spectral, residual_nll, residual_entropy]
        # sort the residuals and return the index
        sorted_residuals = np.argsort(residuals)
        if sorted_residuals[0] == 3  and sorted_residuals[-1] == 0:
            qual = {}
            qual['prompt'] = prompt
            qual['references'] = references[0]
            qual['generated'] = generated
            # qual[r'$\mathbb{P}(A \leq a)$'] = score
            qual[r'$\mathbb{P}(U_{\rm Ecc} \leq u)$'] = 1-ecc
            qual[r'$\mathbb{P}(U_{\rm Deg} \leq u)$'] = 1-degree
            qual[r'$\mathbb{P}(U_{\rm EigV} \leq u)$'] = 1-spectral
            qual[r'$\mathbb{P}(U_{\rm SE} \leq u)$'] = 1-entropy
            qual[r'$\mathbb{P}(U_{\rm NLL} \leq u)$'] = 1-nll
            if len(references) == 1:
                quals.append(qual)

            if len(quals) > 50:
                break
    df = pd.DataFrame(quals)
    for col in [r'$\mathbb{P}(U_{\rm Ecc} \leq u)$', r'$\mathbb{P}(U_{\rm Deg} \leq u)$', r'$\mathbb{P}(U_{\rm EigV} \leq u)$', r'$\mathbb{P}(U_{\rm SE} \leq u)$', r'$\mathbb{P}(U_{\rm NLL} \leq u)$']:
        df[col] = df[col].apply(lambda x: f'{x:.3f}')  
    # print df as latex table
    print(df.to_latex(index=False))

    