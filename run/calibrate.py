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
from tasks import factoid
from models import opensource, gpt
from metrics import correctness
from indicators import whitebox

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--dataset', type=str, default='nq-open')
    parser.add_argument('--split', type=str, default='validation')
    parser.add_argument('--correctness', type=str, default='rouge')
    parser.add_argument('--metric_threshold', type=float, default=0.5)
    parser.add_argument('--mode', type=str, default='rouge1')
    parser.add_argument('--indicator', type=str, default='semantic_entropy')
    parser.add_argument('--seed', type=int, default=42)
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
    NQ_Open = factoid.NQ_Open(pipe.tokenizer, args.split)
    dataset = NQ_Open.get_dataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # collect generation and check correctness
    results = []
    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        if idx > 100:
            break
        prompts = batch['prompt']
        generated = pipe.generate(prompts, max_length=50, do_sample=False, return_full_text=False)[0]
        generations = [opensource.TextGenerationModel.clean_generation(element['generated_text']) for element in generated]
        references = [text[0] for text in batch['answer']]
        result = score(generations, [references])

        generateds = pipe.generate(prompts, max_length=50, num_return_sequences=30, do_sample=True, return_full_text=False)
        
        if args.indicator == 'semantic_entropy':
            se = whitebox.SemanticEntropy(
                prompts=prompts, 
                generateds=generateds, 
                model=pipe.model, 
                tokenizer=pipe.tokenizer, device='cuda')
            entropy = se.compute_scores(normalize=True)
            result['confidence'] = entropy[0].item()
        elif args.indicator == 'perplexity':
            Perplexity = whitebox.PerplexityScore(model=args.model)
            perplexities = Perplexity.compute_scores(generateds)
            result['confidence'] = perplexities[0]['mean_perplexity'].item()
        elif args.indicator == 'generation_probability':
            GenerationProbability = whitebox.GenerationProbability(model=pipe.model, tokenizer=pipe.tokenizer)
            probabilities = GenerationProbability.compute_scores(prompts, [generated])
            result['confidence'] = -probabilities[0]['average_neg_log_likelihoods'].item()
        else:
            raise ValueError(f"Indicator {args.indicator} not supported")
        results.append(result)
    
    # generate pandas dataframe from results
    df = pd.DataFrame(results)
    # model name 
    model = args.model.split('/')[-1]
    # save results to csv
    df.to_csv(f'../tmp/calibrate_{model}_{args.indicator}.csv', index=False)

    print("----------------------------------")
    logging.log(logging.INFO, f"Results saved to ../tmp/calibrate.csv")
    print("----------------------------------")
