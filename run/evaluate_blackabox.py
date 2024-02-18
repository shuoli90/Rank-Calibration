import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import datasets
import argparse
from transformers import set_seed
from transformers.pipelines.pt_utils import KeyDataset
import pandas as pd
from tqdm import tqdm
import logging
from tasks import openbook, closedbook
from models import opensource, gpt
from metrics import correctness
from indicators import whitebox, blackbox
from utils import text_processing

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
    if os.path.exists(f'../tmp/calibrate_{model}_{args.indicator}_{args.dataset}.csv'):
        print("----------------------------------")
        logging.log(logging.INFO, f"Results already saved to ../tmp/calibrate_{model}_{args.indicator}_{args.dataset}.csv")
        print("----------------------------------")
        pd = pd.read_csv(f'../tmp/calibrate_{model}_{args.indicator}_{args.dataset}.csv')
    else:
        raise ValueError(f"Results not found at ../tmp/calibrate_{model}_{args.indicator}_{args.dataset}.csv")

    results = []
    for idx, row in pd.iterrows():
        result = {}
        prompts = row['prompt']
        references = row['references']
        generations = row['generations']

        ecc = blackbox.Eccentricity(affinity_mode='jaccard')
        ecc_u, ecc_c = ecc.compute_scores([prompts], [generations])
        result['ecc_u'] = ecc_u[0]
        result['ecc_c'] = ecc_c[0]

        degree = blackbox.Degree(affinity_mode='jaccard')
        degree_u, degree_c= degree.compute_scores([prompts], [generations])
        result['degree_u'] = degree_u[0]
        result['degree_c'] = degree_c[0]

        spectral = blackbox.SpectralEigv(affinity_mode='jaccard', temperature=None)
        spectral_u = spectral.compute_scores([prompts], [generations])
        result['spectral_u'] = spectral_u[0]
    
    # generate pandas dataframe from results
    df = pd.DataFrame(results)
    # save results to csv
    df.to_csv(f'../tmp/calibrate_{model}_blackbox_{args.dataset}.csv', index=False)

    print("----------------------------------")
    logging.log(logging.INFO, f"Results saved to ../tmp/calibrate_{model}_blackbox_{args.dataset}.csv")
    print("----------------------------------")

