import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from indicators import whitebox
from models import opensource

if __name__ == '__main__':

    pipe = opensource.TextGenerationModel(model_name="facebook/opt-350m", torch_dtype=torch.bfloat16)
    # prompts = ['Once upon a time']
    prompts = ['Once upon a time', 'when the sun rises']
    references = [['Yes, I am'], ['No, I am not']]
    generateds = pipe.generate(prompts, max_length=50, num_return_sequences=5, do_sample=True)
    most_likely_generations = pipe.generate(prompts, max_length=50, num_return_sequences=1, do_sample=False)
    
    # semantic entropy
#     se = whitebox.SemanticEntropy(
#             prompts=prompts, 
#             generateds=generateds, 
#             model=pipe.model, 
#             tokenizer=pipe.tokenizer, device='cuda')
#     entropy = se.compute_scores(normalize=True)

    # perplexity score
#     Perplexity = whitebox.PerplexityScore(model="facebook/opt-350m")
#     perplexities = Perplexity.compute_scores(generateds)

    perplexity = whitebox.PerplexityScore(pipe=pipe)
    result = perplexity.compute_scores(prompts, references)
    
    # generation probability
    GenerationProbability = whitebox.GenerationProbability(model=pipe.model, tokenizer=pipe.tokenizer)
    probabilities = GenerationProbability.compute_scores(prompts, generateds)
    print(probabilities)