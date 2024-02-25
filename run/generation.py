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
from tasks import openbook, closedbook, longform 
from models import opensource, gpt
from metrics import correctness
from utils import text_processing
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model', type=str, default='google/gemma-7b-it')
    # parser.add_argument('--model', type=str, default='mistralai/Mistral-7B-Instruct-v0.2')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--dataset', type=str, default='triviaqa')
    parser.add_argument('--split', type=str, default='validation')
    parser.add_argument('--correctness', type=str, default='rouge')
    parser.add_argument('--metric_threshold', type=float, default=0.5)
    parser.add_argument('--mode', type=str, default='rouge1')
    parser.add_argument('--indicator', type=str, default='semantic_entropy')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--task', type=str, default='text-generation')
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--num_return_sequences", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    print("----------------------------------")
    logging.basicConfig(level=logging.INFO)
    logging.log(logging.INFO, f"Running {args.model} on {args.dataset} {args.split} split")
    print("----------------------------------")

    # set seed
    set_seed(args.seed)

    # setup model and dataset
    if 'gpt' in args.model:
        pipe = gpt.GPTModel(model_name=args.model)
    else:
        pipe = opensource.TextGenerationModel(model_name=args.model, torch_dtype=torch.bfloat16, device=args.device)
        if pipe.tokenizer.pad_token_id is None:
            pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
    if args.dataset == 'nq-open':
        NQ_Open = closedbook.NQ_Open(pipe.tokenizer, args.split)
        dataset = NQ_Open.get_dataset()
    elif args.dataset == 'triviaqa':
        TriviaQA = openbook.TriviaQA()
        dataset = TriviaQA.get_dataset()
    elif args.dataset == 'squad':
        SQuAD = openbook.SQuAD()
        dataset = SQuAD.get_dataset()
    elif args.dataset == 'meadow':
        Meadow = longform.Meadow()
        dataset = Meadow.get_dataset()
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    model = args.model.split('/')[-1]
    # collect generation
    results = []
    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        result = {}
        prompts = batch['prompt']
        generated = pipe.generate(prompts, max_length=1024, do_sample=False, return_full_text=False)[0]
        generations = [text_processing.clean_generation(element['generated_text']) for element in generated]
        references = [text for text in batch['answers']]
        result['id'] = idx
        result['prompt'] = prompts
        result['references'] = references
        result['greedy'] = generations

        generateds = pipe.generate(prompts, max_length=1024, num_return_sequences=10, temperature=1.0, do_sample=True, top_p=0.9, return_full_text=False)
        generations = [text_processing.clean_generation(element['generated_text']) for element in generateds[0]]
        result['sampled'] = generations
        results.append(result)

        if idx % 10 == 0:
            # generate pandas dataframe from results
            df = pd.DataFrame(results)
            json.dump(results, open(f'../collected/{model}_{args.dataset}.json', 'w'))

            print("----------------------------------")
            logging.log(logging.INFO, f"Results saved to ../collected/{model}_{args.dataset}.json")
            print("----------------------------------")        

    # generate pandas dataframe from results
    df = pd.DataFrame(results)
    json.dump(results, open(f'../collected/{model}_{args.dataset}.json', 'w'))

    print("----------------------------------")
    logging.log(logging.INFO, f"Results saved to ../collected/{model}_{args.dataset}.json")
    print("----------------------------------")

