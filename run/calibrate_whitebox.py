import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import argparse
from transformers import set_seed
import logging
from tqdm import tqdm
from indicators import whitebox
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
    if os.path.exists(f'../collected/{model}_{args.dataset}.json'):
        print("----------------------------------")
        logging.log(logging.INFO, f"Results already saved to ../tmp/calibrate_{model}_{args.dataset}.json")
        print("----------------------------------")
        data = json.load(open(f'../collected/{model}_{args.dataset}.json'))
    else:
        raise ValueError(f"Results not found at ../collected/{model}_{args.dataset}.json")

    SE = whitebox.SemanticEntropy(
        device='cuda')
    
    results = []
    for idx, row in tqdm(enumerate(data), total=len(data)):
        result = {}
        prompt = row['prompt']
        references = row['references']
        generations = row['generated']
        transition_score = [np.array(score) for score in row['transition_score']] # log probability of each token

        # negative loglikelihood
        unnormalized_nll = [-np.sum(lls[np.isfinite(lls)]) for lls in transition_score]
        normalized_nll = [-np.mean(lls[np.isfinite(lls)]) for lls in transition_score]
        result['unnormalized_nll'] = unnormalized_nll
        result['normalized_nll'] = normalized_nll

        # semantic entropy
        entropy_unnormalized = SE.compute_scores([prompt], [generations], nlls=[unnormalized_nll])
        result['entropy_unnormalized'] = entropy_unnormalized[0].item()
        entropy_normalized = SE.compute_scores([prompt], [generations], nlls=[normalized_nll])
        result['entropy_normalized'] = entropy_normalized[0].item()
        results.append(result)

        if idx % 10 == 0:
            json.dump(results, open(f'../tmp/calibrate_{model}_{args.dataset}_whitebox.json', 'w'))
    # generate pandas dataframe from results
    json.dump(results, open(f'../tmp/calibrate_{model}_{args.dataset}_whitebox.json', 'w'))

    print("----------------------------------")
    logging.log(logging.INFO, f"Results saved to ../tmp/calibrate_{model}_{args.dataset}_whitebox.csv")
    print("----------------------------------")

