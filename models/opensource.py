import torch
from transformers import pipeline
import functools
import numpy as np

class TextGenerationModel:
    
    def __init__(self, model_name='meta-llama/Llama-2-7b-hf', **kwargs):
        self.pip = pipeline(model=model_name, device_map="auto", **kwargs)

    def generate(self, prompts, **kwargs):
        return self.pip(prompts, **kwargs)
    
    @functools.cached_property
    def model(self):
        return self.pip.model
    
    @functools.cached_property
    def tokenizer(self):
        return self.pip.tokenizer
    
    @classmethod
    def clean_generation(cls, generation):
        strings_to_filter_on = [
                    '.', '\n', 'Q:', 'A:', 'question:', 'answer:', 'Question:', 'Answer:', 'Questions:', 'questions:', 'QUESTION:',
                    'ANSWER:'
                ]
        for string in strings_to_filter_on:
            if string in generation:
                generation = generation.split(string)[0]
        return generation.strip()

class NLIModel:
    
    def __init__(self, model_name='microsoft/deberta-large-mnli', **kwargs):
        self.pipe = pipeline(model=model_name, **kwargs)
    
    @torch.no_grad()
    def classify(self, question, prompts, **kwargs):
        # https://github.com/lorenzkuhn/semantic_uncertainty
        # https://github.com/zlin7/UQ-NLG
        semantic_set_ids = {ans: i for i, ans in enumerate(prompts)}
        _rev_mapping = semantic_set_ids.copy()
        sim_mat_batch = torch.zeros((len(prompts), len(prompts),3))
        make_input = lambda x: dict(text=x[0],text_pair=x[1])
        for i, ans_i in enumerate(prompts):
            for j, ans_j in enumerate(prompts):
                # if i == j: continue # may not needed
                scores = self.pipe(make_input([f"{question} {ans_i}", f"{question} {ans_j}"]), return_all_scores=True, **kwargs)
                sim_mat_batch[i,j] = torch.tensor([score['score'] for score in scores])
        return dict(
            mapping = [_rev_mapping[_] for _ in prompts],
            sim_mat = sim_mat_batch
        )
    
    @torch.no_grad()
    def compare(self, question, ans_1, ans_2, **kwargs):
        prompt1 = dict(text=f'{question} {ans_1}', text_pair=f'{question} {ans_2}')
        prompt2 = dict(text=f'{question} {ans_2}', text_pair=f'{question} {ans_1}')
        logits_list = self.pip([prompt1, prompt2], return_all_scores=True, **kwargs)
        logits = torch.tensor([[logit['score'] for logit in logits] for logits in logits_list])
        pred = 0 if logits.argmax(dim=1).min() == 0 else 1
        return {
            'deberta_prediction': pred,
            'prob': logits.cpu(),
            'pred': logits.cpu()
        }
