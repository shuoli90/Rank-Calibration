import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from indicators import whitebox
from models import opensource

if __name__ == '__main__':

    pipe = opensource.TextGenerationModel(model_name=f"facebook/opt-350m", torch_dtype=torch.bfloat16)
    # prompts = ['Once upon a time']
    prompts = ['Once upon a time', 'when the sun rises']
    generateds = pipe.generate(prompts, max_length=50, num_return_sequences=5, do_sample=True)
    most_likely_generations = pipe.generate(prompts, max_length=50, num_return_sequences=1, do_sample=False)
    se = whitebox.SemanticEntropy(
        prompts=prompts, 
        generateds=generateds, 
        most_likely_generations=most_likely_generations, 
        model=pipe.model, 
        tokenizer=pipe.tokenizer, device='cuda')
    similarities = se.similarities
    neg_log_likelihoods = se.neg_log_likelihoods
    entropy = se.compute_scores(normalize=True)