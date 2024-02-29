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
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--dataset', type=str, default='triviaqa')
    parser.add_argument('--split', type=str, default='validation')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--num_return_sequences", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument('--device', type=int, default=7)
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
    
    def collate_fn(data):
        prompts = [example['prompt'] for example in data]
        answers = [example['answers'] for example in data]
        return prompts, answers

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    model = args.model.split('/')[-1]
    results = []
    for idx, (prompts, answers) in tqdm(enumerate(dataloader), total=len(dataloader)):
        if args.dataset == 'meadow' and idx > 1000:
            break
        generateds, transition_scores = pipe.generate(
                                    prompts, max_length=args.max_length, 
                                    num_return_sequences=args.num_return_sequences, 
                                    temperature=args.temperature, top_p=args.top_p, 
                                    do_sample=True,
                                    return_dict_in_generate=True, output_scores=True,
                                    repetition_penalty=100.0)
        if len(prompts) == 1:
            generateds_list = [generateds]
            transition_scores = [transition_scores]
        generateds = [[text_processing.clean_generation(generated.split('A: ')[-1]) for generated in generateds] for generateds in generateds_list]
        tmp = [{'prompt':prompt, 'references':answer, 'generated':generated, 'transition_score': transition_score.detach().cpu().numpy().tolist()} 
               for prompt, answer, generated, transition_score in zip(prompts, answers, generateds, transition_scores)]
        
        results.extend(tmp)

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

