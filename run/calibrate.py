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

    # setup generation correctness score
    score = correctness.Score(metric_name=args.correctness, mode=args.mode, metric_threshold=args.metric_threshold)
    # setup model and dataset
    if 'gpt' in args.model:
        pipe = gpt.GPTModel(model_name=args.model)
    else:
        pipe = opensource.TextGenerationModel(model_name=args.model, torch_dtype=torch.bfloat16)
        if pipe.tokenizer.pad_token_id is None:
            pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
    if args.dataset == 'nq-open':
        NQ_Open = closedbook.NQ_Open(pipe.tokenizer, args.split)
        dataset = NQ_Open.get_dataset()
    elif args.dataset == 'triviaqa':
        TriviaQA = openbook.TriviaQA()
        dataset = TriviaQA.get_dataset()
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    GenerationProbability = whitebox.GenerationProbability(pipe=pipe)
    if args.indicator == 'semantic_entropy':
        se = whitebox.SemanticEntropy(
            pipe=pipe, 
            device='cuda')
    elif args.indicator == 'self_consistency':
        SC =  blackbox.SelfConsistencyConfidence(pipe=pipe)
    else:
        raise ValueError(f"Indicator {args.indicator} not supported")
    
    # collect generation and check correctness
    results = []
    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        result = {}
        if idx > 3:
            break
        prompts = batch['prompt']
        generated = pipe.generate(prompts, max_length=256, do_sample=False, return_full_text=False)[0]
        generations = [text_processing.clean_generation(element['generated_text']) for element in generated]
        references = [text[0] for text in batch['answer']]
        result['prompt'] = prompts
        result['references'] = references
        result['greedy'] = generations
        probabilities = GenerationProbability.compute_scores(prompts, [generations])
        result['normalized_nll'] = -probabilities[0]['average_neg_log_likelihoods'].item()
        result['unnormalized_nll'] = -probabilities[0]['neg_log_likelihoods'].item()
        
        generateds = pipe.generate(prompts, max_length=256, num_return_sequences=30, do_sample=True, return_full_text=False)
        # breakpoint()
        generations = [text_processing.clean_generation(element['generated_text']) for element in generateds[0]]
        result['sampled'] = generations
        if args.indicator == 'semantic_entropy':
            entropy = se.compute_scores(prompts, [generations], normalize=False)
            result['entropy'] = entropy[0].item()
        elif args.indicator == 'self_consistency':
            sc = SC.compute_scores(prompts, [generations])
            breakpoint()
            result['self_consistency'] = sc[0].item()
        else:
            raise ValueError(f"Indicator {args.indicator} not supported")
    
    # generate pandas dataframe from results
    df = pd.DataFrame(results)
    # model name 
    model = args.model.split('/')[-1]
    # save results to csv
    df.to_csv(f'../tmp/calibrate_{model}_{args.indicator}.csv', index=False)

    print("----------------------------------")
    logging.log(logging.INFO, f"Results saved to ../tmp/calibrate_{model}_{args.indicator}_{args.dataset}.csv")
    print("----------------------------------")

