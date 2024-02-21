import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
from transformers import set_seed
import numpy as np
from tqdm import tqdm
import logging
from indicators import blackbox
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    # parser.add_argument('--model', type=str, default='facebook/opt-350m')
    parser.add_argument('--dataset', type=str, default='triviaqa')
    parser.add_argument('--split', type=str, default='validation')
    parser.add_argument('--correctness', type=str, default='rouge')
    parser.add_argument('--metric_threshold', type=float, default=0.5)
    parser.add_argument('--mode', type=str, default='rouge1')
    parser.add_argument('--indicator', type=str, default='semantic_entropy')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--task', type=str, default='text-generation')
    args = parser.parse_args()

    print("----------------------------------")
    logging.basicConfig(level=logging.INFO)
    logging.log(logging.INFO, f"Running {args.model} on {args.dataset} {args.split} split")
    logging.log(logging.INFO, f"Using {args.correctness} with threshold {args.metric_threshold} and mode {args.mode}")
    print("----------------------------------")

    # set seed
    set_seed(args.seed)
    model = args.model.split('/')[-1]
    # load collected data
    if os.path.exists(f'../collected/{model}_{args.dataset}.csv'):
        print("----------------------------------")
        logging.log(logging.INFO, f"Results already saved to ../tmp/calibrate_{model}_{args.dataset}.csv")
        print("----------------------------------")
        data = json.load(open(f'../collected/{model}_{args.dataset}.json'))
    else:
        raise ValueError(f"Results not found at ../collected/{model}_{args.dataset}.csv")
    
    ECC = blackbox.Eccentricity(affinity_mode='disagreement')
    DEGREE = blackbox.Degree(affinity_mode='disagreement')
    SPECTRAL = blackbox.SpectralEigv(affinity_mode='disagreement')

    results = []
    for idx, row in tqdm(enumerate(data), total=len(data)):
        result = {"idx": idx}
        prompts = row['prompt']
        references = row['references']
        generations_greedy = row['greedy'] 
        generations_sampled = row['sampled']

        ecc_u, ecc_c = ECC.compute_scores([prompts], [generations_sampled])
        result['ecc_u'] = np.float64(ecc_u[0])
        result['ecc_c'] = ecc_c[0].tolist()
        result['ecc_c'] = [np.float64(x) for x in result['ecc_c']]

        degree_u, degree_c= DEGREE.compute_scores([prompts], [generations_sampled])
        result['degree_u'] = np.float64(degree_u[0])
        result['degree_c'] = degree_c[0].tolist()
        result['degree_c'] = [np.float64(x) for x in result['degree_c']]

        spectral_u = SPECTRAL.compute_scores([prompts], [generations_sampled])
        result['spectral_u'] = np.float64(spectral_u[0])

        results.append(result)
        if idx % 10 == 0:
            json.dump(results, open(f'../tmp/calibrate_{model}_{args.dataset}_blackbox.json', 'w'))
    
    json.dump(results, open(f'../tmp/calibrate_{model}_{args.dataset}_blackbox.json', 'w'))

    print("----------------------------------")
    logging.log(logging.INFO, f"Results saved to ../tmp/calibrate_{model}_{args.dataset}_blackbox.csv")
    print("----------------------------------")

