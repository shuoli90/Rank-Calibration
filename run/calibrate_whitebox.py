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
from models import opensource, gpt
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
    if os.path.exists(f'../collected/{model}_{args.dataset}.csv'):
        print("----------------------------------")
        logging.log(logging.INFO, f"Results already saved to ../tmp/calibrate_{model}_{args.dataset}.csv")
        print("----------------------------------")
        data = json.load(open(f'../collected/{model}_{args.dataset}.json'))
    else:
        raise ValueError(f"Results not found at ../collected/{model}_{args.dataset}.csv")

    if 'gpt' in args.model:
        pipe = gpt.GPTModel(model_name=args.model)
    else:
        pipe = opensource.TextGenerationModel(model_name=args.model, torch_dtype=torch.bfloat16)
        if pipe.tokenizer.pad_token_id is None:
            pipe.tokenizer.pad_token = pipe.tokenizer.eos_token

    GenerationProbability = whitebox.GenerationProbability(pipe=pipe)
    SE = whitebox.SemanticEntropy(
        pipe=pipe, 
        device='cuda')
    
    results = []
    for idx, row in enumerate(data):
        result = {}
        prompts = row['prompt']
        references = row['references']
        generations_greedy = row['greedy']
        generations_sampled = row['sampled']

        # log probability
        probabilities = GenerationProbability.compute_scores(prompts, [generations_greedy])
        result['normalized_nll'] = -probabilities[0]['average_neg_log_likelihoods'].item()
        result['unnormalized_nll'] = -probabilities[0]['neg_log_likelihoods'].item()
        # semantic entropy
        entropy = SE.compute_scores(prompts, [generations_sampled], normalize=False)
        result['entropy'] = entropy[0].item()
        results.append(result)

        if idx % 10 == 0:
            # generate pandas dataframe from results
            df = pd.DataFrame(results)
            # save results to csv
            df.to_csv(f'../tmp/calibrate_{model}_{args.indicator}_whitebox.csv', index=False)

    # generate pandas dataframe from results
    df = pd.DataFrame(results)
    # save results to csv
    df.to_csv(f'../tmp/calibrate_{model}_{args.indicator}_whitebox.csv', index=False)

    print("----------------------------------")
    logging.log(logging.INFO, f"Results saved to ../tmp/calibrate_{model}_{args.dataset}_whitebox.csv")
    print("----------------------------------")

