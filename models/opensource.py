import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
import functools
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class TextGenerationModel:
    
    def __init__(self, model_name='meta-llama/Llama-2-7b-hf', device=7,**kwargs):
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16,).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device
        self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id
        self.tokenizer.pad_token = self.tokenizer.eos_token
        kwargs = {**dict(model=model_name, device=device, use_fast=True), **kwargs}
    
    @torch.no_grad()
    def generate(self, prompts, **kwargs):
        prompts = self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True).to(self.device)
        outputs = self.model.generate(**prompts, **kwargs)
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences,
            outputs.scores,
            normalize_logits=True)
        return self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True), transition_scores

class NLIModel:
    
    def __init__(self, model_name='microsoft/deberta-large-mnli', **kwargs):
        self.pipe = pipeline(model=model_name, **kwargs)
    
    @torch.no_grad()
    def classify(self, prompt, responses, **kwargs):
        '''
        Input:
            prompt: a string-formatted prompt p
            responses: responses [r_1, ..., r_n]
        Output:
            a dictionary contains a mapping of response indexing and a n*n similarity matrix
        '''
        # https://github.com/lorenzkuhn/semantic_uncertainty
        # https://github.com/zlin7/UQ-NLG
        semantic_set_ids = {resp: i for i, resp in enumerate(responses)}
        _rev_mapping = semantic_set_ids.copy()
        sim_mat_batch = torch.zeros((len(responses), len(responses),3))
        make_input = lambda x: dict(text=x[0],text_pair=x[1])
        for i, response_i in enumerate(responses):
            for j, response_j in enumerate(responses):
                # if i == j: continue # may not needed
                scores = self.pipe(make_input([f"{prompt} {response_i}", f"{prompt} {response_j}"]), return_all_scores=True, **kwargs)
                sim_mat_batch[i,j] = torch.tensor([score['score'] for score in scores])
        return dict(
            mapping = [_rev_mapping[_] for _ in responses],
            sim_mat = sim_mat_batch
        )
    
    @torch.no_grad()
    def compare(self, prompt, response_1, response_2, **kwargs):
        prompt1 = dict(text=f'{prompt} {response_1}', text_pair=f'{prompt} {response_2}')
        prompt2 = dict(text=f'{prompt} {response_2}', text_pair=f'{prompt} {response_1}')
        logits_list = self.pipe([prompt1, prompt2], return_all_scores=True, **kwargs)
        logits = torch.tensor([[logit['score'] for logit in logits] for logits in logits_list])
        pred = 0 if logits.argmax(dim=1).min() == 0 else 1
        return {
            'deberta_prediction': pred,
            'prob': logits.cpu(),
            'pred': logits.cpu()
        }

class BERTEmbedding:
    def __init__(self, model_name='sentence-transformers/bert-base-nli-mean-tokens', **kwargs):
        self.pipe = SentenceTransformer(model_name, **kwargs)
    
    @torch.no_grad()
    def compare(self, question, ans_1, ans_2, **kwargs):
        prompt1 = [dict(text=f'{question} {_}') for _ in ans_1]
        prompt2 = [dict(text=f'{question} {_}') for _ in ans_2]
        emb1 = self.pipe.encode(prompt1, **kwargs)
        emb2 = self.pipe.encode(prompt2, **kwargs)
        return np.diag(cosine_similarity(emb1, emb2))