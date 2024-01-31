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
        # return string before the first "\n" character
        idx = np.char.find(generation, "\n", start=0, end=None)
        return generation.strip() if idx == -1 else generation[:idx].strip()

class NLIModel:
    
    def __init__(self, model_name='microsoft/deberta-large-mnli', **kwargs):
        self.pip = pipeline(model=model_name, **kwargs)
    
    @torch.no_grad()
    def classify(self, question, prompts, **kwargs):
        # https://github.com/lorenzkuhn/semantic_uncertainty
        # https://github.com/zlin7/UQ-NLG
        unique_ans = sorted(list(set(prompts)))
        semantic_set_ids = {ans: i for i, ans in enumerate(unique_ans)}
        _rev_mapping = semantic_set_ids.copy()
        sim_mat_batch = torch.zeros((len(unique_ans), len(unique_ans),3))
        pairs = []
        make_input = lambda x: dict(text=x[0],text_pair=x[1])
        for i, ans_i in enumerate(unique_ans):
            for j, ans_j in enumerate(unique_ans):
                if i == j: continue
                logits = self.pip(make_input([f"{question} {ans_i}", f"{question} {ans_j}"]), return_all_scores=True, **kwargs)
                sim_mat_batch[i,j] = torch.tensor([logit['score'] for logit in logits])
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
