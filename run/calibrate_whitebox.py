import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import argparse
from transformers import set_seed
import logging
from tqdm import tqdm
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
    if os.path.exists(f'../collected/{model}_{args.dataset}_new.json'):
        print("----------------------------------")
        logging.log(logging.INFO, f"Results already saved to ../tmp/calibrate_{model}_{args.dataset}.json")
        print("----------------------------------")
        data = json.load(open(f'../collected/{model}_{args.dataset}_new.json'))
    else:
        raise ValueError(f"Results not found at ../collected/{model}_{args.dataset}_new.json")

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
    for idx, row in tqdm(enumerate(data), total=len(data)):
        result = {'idx':idx}
        prompts = row['prompt']
        references = row['references']
        generations_greedy = row['greedy']
        generations_sampled = row['sampled']

        # log probability
        probabilities = GenerationProbability.compute_scores(prompts, [generations_greedy])
        result['normalized_nll'] = probabilities[0]['average_neg_log_likelihoods'].item()
        result['unnormalized_nll'] = probabilities[0]['neg_log_likelihoods'].item()
        # semantic entropy
        entropy = SE.compute_scores(prompts, [generations_sampled], normalize=False)
        result['entropy'] = entropy[0].item()
        results.append(result)

        if idx % 10 == 0:
            json.dump(results, open(f'../tmp/calibrate_{model}_{args.dataset}_whitebox.json', 'w'))
    # generate pandas dataframe from results
    json.dump(results, open(f'../tmp/calibrate_{model}_{args.dataset}_whitebox.json', 'w'))

    print("----------------------------------")
    logging.log(logging.INFO, f"Results saved to ../tmp/calibrate_{model}_{args.dataset}_whitebox.csv")
    print("----------------------------------")

