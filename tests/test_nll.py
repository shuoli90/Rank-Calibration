import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from models import opensource
from indicators.whitebox import get_neg_loglikelihoods
from transformers import AutoTokenizer, AutoModelForCausalLM

if __name__ == '__main__':
    # model = AutoModelForCausalLM.from_pretrained(f"facebook/opt-350m",
    #                                          torch_dtype=torch.float16).cuda()
    # tokenizer = AutoTokenizer.from_pretrained(f"facebook/opt-350m",
    #                                         use_fast=False)
    
    pipe = opensource.TextGenerationModel(model_name=f"facebook/opt-350m", torch_dtype=torch.bfloat16)
    prompt = 'Once upon a time'
    generated = pipe.generate(prompt, max_length=50, num_return_sequences=5, do_sample=True)
    most_likely_generation = pipe.generate(prompt, max_length=50, num_return_sequences=1, do_sample=False)
    sequences = [{
        'prompt': torch.tensor(pipe.tokenizer.encode(prompt)),
        'generations': torch.tensor([pipe.tokenizer.encode(generated[i]['generated_text'], truncation=True, max_length=42) for i in range(len(generated))]),
        'most_likely_generation_ids': torch.tensor(pipe.tokenizer.encode(most_likely_generation[0]['generated_text'], truncation=True, max_length=42)),
        'id': 0
    }]

    results = get_neg_loglikelihoods(pipe.model, pipe.tokenizer, sequences)