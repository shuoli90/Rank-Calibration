import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
from transformers import set_seed
import numpy as np
from tqdm import tqdm
import logging
from indicators import blackbox
from models import gpt
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo')
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument('--dataset', type=str, default='triviaqa')
    parser.add_argument('--split', type=str, default='validation')
    parser.add_argument('--mode', type=str, default='rouge1')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument('--affinity_mode', type=str, default='disagreement')
    args = parser.parse_args()

    print("----------------------------------")
    logging.basicConfig(level=logging.INFO)
    logging.log(logging.INFO, f"Running {args.model} on {args.dataset} {args.split} split")
    print("----------------------------------")

    # set seed
    set_seed(args.seed)
    model_name = args.model.split('/')[-1]
    # load collected data
    if os.path.exists(f'../collected/{model_name}_{args.dataset}_{args.temperature}.json'):
        print("----------------------------------")
        logging.log(logging.INFO, f"Results already saved to ../tmp/calibrate_{model_name}_{args.dataset}_{args.temperature}.json")
        print("----------------------------------")
        data = json.load(open(f'../collected/{model_name}_{args.dataset}_{args.temperature}.json'))
    else:
        raise ValueError(f"Results not found at ../collected/{model_name}_{args.dataset}_{args.temperature}.json")
    
    model = gpt.GPTModel()
    verbalized = blackbox.VerbalizedConfidence(pipe=model)

    if os.path.exists(f'../tmp/calibrate_{model_name}_{args.dataset}_{args.temperature}_{args.affinity_mode}_verbalized.json'):
        results = json.load(open(f'../tmp/calibrate_{model_name}_{args.dataset}_{args.temperature}_{args.affinity_mode}_verbalized.json'))
    else:
        results = []
    collected_length = len(results)
    for idx, row in tqdm(enumerate(data), total=len(data)):
        if idx < collected_length:
            continue
        try:
            result = {"idx": idx}
            prompts = row['prompt']
            references = row['references']
            generations_sampled = row['generated']

            question = prompts.split('\n')[-2].strip()
            verbalized_score = verbalized.compute_scores([question], [generations_sampled])
            result['verbalized'] = verbalized_score[0]

            results.append(result)
            if idx % 10 == 0:
                json.dump(results, open(f'../tmp/calibrate_{model_name}_{args.dataset}_{args.temperature}_{args.affinity_mode}_verbalized.json', 'w'))
        except:
            print(f"Error at idx {idx}")
            continue
    
    json.dump(results, open(f'../tmp/calibrate_{model_name}_{args.dataset}_{args.temperature}_{args.affinity_mode}_verbalized.json', 'w'))

    print("----------------------------------")
    logging.log(logging.INFO, f"Results saved to ../tmp/calibrate_{model_name}_{args.dataset}_{args.temperature}_{args.affinity_mode}_verbalized.csv")
    print("----------------------------------")