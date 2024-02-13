import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from indicators import whitebox
from models import opensource

if __name__ == '__main__':

    pipe = opensource.TextGenerationModel(model_name="meta-llama/Llama-2-7b-hf", torch_dtype=torch.bfloat16)
    # prompts = ['Once upon a time']
    prompts = ['Once upon a time', 'when the sun rises']

    references = [['Yes, I am', 'Good morning'], ['No, I am not', 'see you soon']]

    # generations = ['Leukemia', 'Low Blood Pressure', 'Cervical cancer', 'Cancer in 1953 at 41', 'Breast cancer', 'Tuberculosis', 'Cancer', 'Leukaemia',
                #    'Cancer (in 1953 at age 41)', 'Throat cancer']
    # generations = ['Pink Floyd', 'Pink Floyd in Edinburgh', 'Pink Floyd', 'Shambles', 'Pink Floyd', 'Pink Floyd', 'Pink Floyd', 'Pink Floyd', 'Pink Floyd', 'Pink Floyd']
    generations = ['Pink Floyd', 'Pink Floyd', 'Pink Floyd', 'Pink Floyd', 'Pink Floyd', 'Pink Floyd', 'Pink Floyd', 'Pink Floyd', 'Pink Floyd', 'Pink Floyd']
    # generateds = pipe.generate(prompts, max_length=50, num_return_sequences=5, do_sample=True, return_full_text=False)
    # gen_texts = [[opensource.TextGenerationModel.clean_generation(gen['generated_text']) for gen in generated] for generated in generateds]
    # most_likely_generations = pipe.generate(prompts, max_length=50, num_return_sequences=1, do_sample=False, return_full_text=False)
    
    # semantic entropy
    se = whitebox.SemanticEntropy(pipe=pipe)
    entropy = se.compute_scores(["Q: Dave Gilmore and Roger Waters were in which rock group? A:"], [generations])
    print(entropy)

    # perplexity score
#     Perplexity = whitebox.PerplexityScore(model="facebook/opt-350m")
#     perplexities = Perplexity.compute_scores(generateds)

    perplexity = whitebox.PerplexityScore(pipe=pipe)
    result = perplexity.compute_scores(prompts, references)
    # perplexity = whitebox.PerplexityScore(pipe=pipe)
    # result = perplexity.compute_scores(prompts, references)
    # breakpoint()
    
    # generation probability
    # GenerationProbability = whitebox.GenerationProbability(pipe=pipe)
    # probabilities = GenerationProbability.compute_scores(prompts, gen_texts)
    # print(probabilities)