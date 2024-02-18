import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from models import opensource

if __name__ == '__main__':
    model = opensource.TextGenerationModel(model_name='meta-llama/Llama-2-7b-hf', torch_dtype=torch.bfloat16)
    prompt = 'Once upon a time'

    input_ids = model.tokenizer(prompt, return_tensors='pt').input_ids
    generated = model.model.generate(input_ids, max_length=50, num_return_sequences=1, 
                                     return_dict_in_generate=True, output_scores=True)
    transition_scores = model.model.compute_transition_scores(
        generated.sequences, 
        generated.scores, 
        normalize_logits=True
    )
    breakpoint()

    