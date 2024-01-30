import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import datasets
import argparse
from transformers.pipelines.pt_utils import KeyDataset
import pandas as pd
from tqdm import tqdm
import logging
from tasks import opendomain
from models import opensource
from metrics import correctness
from indicators import whitebox

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="facebook/opt-350m")
    parser.add_argument('--dataset', type=str, default='nq-open')
    parser.add_argument('--split', type=str, default='validation')
    parser.add_argument('--correctness', type=str, default='rouge')
    parser.add_argument('--metric_threshold', type=float, default=0.5)
    parser.add_argument('--mode', type=str, default='rouge1')
    args = parser.parse_args()

    print("----------------------------------")
    logging.basicConfig(level=logging.INFO)
    logging.log(logging.INFO, f"Running {args.model} on {args.dataset} {args.split} split")
    logging.log(logging.INFO, f"Using {args.correctness} with threshold {args.metric_threshold} and mode {args.mode}")
    print("----------------------------------")

    # setup generation correctness score
    score = correctness.Score(metric_name=args.correctness, mode=args.mode, metric_threshold=args.metric_threshold)
    # setup model and dataset
    pipe = opensource.TextGenerationModel(model_name=args.model, torch_dtype=torch.bfloat16)
    NQ_Open = opendomain.NQ_Open(pipe.tokenizer, args.split)
    dataset = NQ_Open.get_dataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # collect generation and check correctness
    results = []
    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        if idx > 4:
            break
        prompts = batch['prompt']
        generated = pipe.generate(prompts, max_length=50, do_sample=False, return_full_text=False)
        generations = [element[0]['generated_text'] for element in generated]
        references = [text[0] for text in batch['answer']]
        result = score(generations, [references])

        generateds = pipe.generate(prompts, max_length=50, num_return_sequences=5, do_sample=True)
        se = whitebox.SemanticEntropy(
            prompts=prompts, 
            generateds=generateds, 
            model=pipe.model, 
            tokenizer=pipe.tokenizer, device='cuda')
        entropy = se.compute_scores(normalize=True)
        result['confidence'] = entropy[0].item()
        results.append(result)
    
    # generate pandas dataframe from results
    df = pd.DataFrame(results)
    # compute stats
    breakpoint()
