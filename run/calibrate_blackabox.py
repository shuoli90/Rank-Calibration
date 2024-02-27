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
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument('--affinity_mode', type=str, default='disagreement')
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
    if os.path.exists(f'../collected/{model}_{args.dataset}.json'):
        print("----------------------------------")
        logging.log(logging.INFO, f"Results already saved to ../tmp/calibrate_{model}_{args.dataset}.json")
        print("----------------------------------")
        data = json.load(open(f'../collected/{model}_{args.dataset}.json'))
    else:
        raise ValueError(f"Results not found at ../collected/{model}_{args.dataset}.json")
    
    ECC = blackbox.Eccentricity(affinity_mode=args.affinity_mode, device=args.device)
    DEGREE = blackbox.Degree(affinity_mode=args.affinity_mode, device=args.device, semantic_model=ECC.sm)
    SPECTRAL = blackbox.SpectralEigv(affinity_mode=args.affinity_mode, device=args.device, semantic_model=ECC.sm)

    results = []
    for idx, row in tqdm(enumerate(data), total=len(data)):
        result = {"idx": idx}
        prompts = row['prompt']
        references = row['references']
        generations_greedy = row['greedy'] 
        generations_sampled = row['sampled']

        # split out the last question
        prompts = ['Question: ' + prompt.split('Question: ')[-1] for prompt in prompts]

        # ecc_u, ecc_c, sim_mats = ECC.compute_scores([prompts], [generations_sampled])
        ecc_u, ecc_c, sim_mats = ECC.compute_scores([[""]], [generations_sampled])
        result['ecc_u'] = np.float64(ecc_u[0])
        result['ecc_c'] = ecc_c[0].tolist()
        result['ecc_c'] = [np.float64(x) for x in result['ecc_c']]

        degree_u, degree_c, sim_mats = DEGREE.compute_scores([prompts], [generations_sampled], batch_sim_mats=sim_mats)
        result['degree_u'] = np.float64(degree_u[0])
        result['degree_c'] = degree_c[0].tolist()
        result['degree_c'] = [np.float64(x) for x in result['degree_c']]

        spectral_u, spectral_c, sim_mats = SPECTRAL.compute_scores([prompts], [generations_sampled], sim_mats=sim_mats)
        result['spectral_u'] = np.float64(spectral_u[0])

        results.append(result)
        if idx % 10 == 0:
            json.dump(results, open(f'../tmp/calibrate_{model}_{args.dataset}_{args.affinity_mode}_blackbox.json', 'w'))
    
    json.dump(results, open(f'../tmp/calibrate_{model}_{args.dataset}_{args.affinity_mode}_blackbox.json', 'w'))

    print("----------------------------------")
    logging.log(logging.INFO, f"Results saved to ../tmp/calibrate_{model}_{args.dataset}_{args.affinity_mode}_blackbox.csv")
    print("----------------------------------")

