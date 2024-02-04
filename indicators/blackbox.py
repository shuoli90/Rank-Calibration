import torch
import functools
import evaluate
import numpy as np
from itertools import permutations
from models.opensource import TextGenerationModel
import re

def demo_perturb(demos):
    demo_list = list(permutations(demos))
    return [[*demo] for demo in demo_list]

class BlackBox():

    def __init__(self):
        return NotImplementedError

    def compute_scores(self):
        return NotImplementedError

class ICLRobust():
    def __init__(self, pipe, demo_transforms=None):
        self.pipe = pipe
        self.demo_transforms = demo_transforms
    
    def generate(self, demonstrations, prompt, **kwargs):
        if self.demo_transforms is not None:
            demonstrations_list = self.demo_transforms(demonstrations)
        else:
            demonstrations_list = [demonstrations]
        prompts = [" ".join([*demo, prompt]) for demo in demonstrations_list]
        generations = [self.pipe.generate(prompt_tmp, **kwargs) for prompt_tmp in prompts]
        return generations
    
    def compute_scores(self, consistency_meansure, demonstrations, prompt, **kwargs):
        generations = self.generate(demonstrations, prompt, **kwargs)
        return consistency_meansure(generations)

class ReparaphraseRobust():

    def __init__(self, pipe, prompt_transforms=None):
        self.pipe = pipe
        self.prompt_transforms = prompt_transforms
    
    def generate(self, prompt, **kwargs):
        if self.prompt_transforms is not None:
            prompt_list = self.prompt_transforms(prompt)
        else:
            prompt_list = [prompt]
        generations = [self.pipe.generate(prompt, **kwargs) for prompt in prompt_list]
        return generations

class SelfConsistency(BlackBox):
    def __init__(self, pipe, score_name='exact_match'):
        self.pipe = pipe
        self.score_name = score_name
        self.score = evaluate.load(score_name)

    def compute_scores(self, prompt, gen_text, num_add_trials=5, **kwargs):
        # for a single (query, gen_text) pair
        re_generateds = self.pipe.generate(prompt, num_return_sequences=num_add_trials, max_length=50, do_sample=True, return_full_text=False)
        re_gen_texts = [TextGenerationModel.clean_generation(re_generated['generated_text']) for re_generated in re_generateds]
        scores = self.score.compute(references=[gen_text]*num_add_trials, predictions=re_gen_texts)[self.score_name]
        return np.mean(scores)

class Verbalized(BlackBox):
    def __init__(self, pipe=None, model=None):
        self.pipe = pipe
        self.model = model
        self.tokenizer = self.pipe.tokenizer if self.pipe else self.model.tokenizer
        self.description1 = "Read the question and answer.\n"
        self.description2 = "\nProvide a numeric confidence that indicates your certainty about this answer. For instance, if your confidence level is 80%, it means you are 80% certain that this answer is correct and there is a 20% chance that it is incorrect. Use the following format to provide your confidence: Confidence: [Your confidence, a numerical number in the range of 0-100]%."

    def extract_confidence(self, s):
        pattern = r'Confidence:\s*(\d+)%'
        match = re.findall(pattern, s)
        if match:
            conf = int(match[0])/100
            return conf
        else:
            raise ValueError("No formatted verbalized confidence available!")

    def compute_scores(self, prompt, gen_text, **kwargs):
        # for a single (query, gen_text) pair
        combo_text = self.description1+prompt+gen_text+self.description2
        cur_length = len(self.tokenizer(combo_text)['input_ids'])
        if self.pipe:
            verbal_conf = self.pipe.generate(combo_text, max_length=cur_length+10, return_full_text=False)[0]['generated_text']
        elif self.model:
            # for GPT APIs
            verbal_conf = self.model.generate([combo_text], max_token=10)[0]['generated_text']
        else:
            raise ValueError("Please specify a valid pipeline or model!")
        return self.extract_confidence(verbal_conf)

class Hybrid(BlackBox):
    # https://arxiv.org/pdf/2306.13063.pdf
    # use self consistency emsemble with verbalized confidences
    def __init__(self, pipe=None, model=None, score_name='exact_match'):
        self.pipe = pipe
        self.model = model
        self.score_name = score_name
        self.score = evaluate.load(score_name)
        self.vb = Verbalized(pipe=pipe, model=model)

    def compute_scores(self, prompt, gen_text, num_add_trials=5, **kwargs):
        # for a single (query, gen_text) pair
        gen_conf = self.vb.compute_scores(prompt, gen_text)
        if self.pipe:
            re_generateds = self.pipe.generate(prompt, num_return_sequences=num_add_trials, max_length=50, do_sample=True, return_full_text=False)
        elif self.model:
            # for GPT APIs
            re_generateds = self.model.generate([prompt], num_return_sequences=num_add_trials, max_tokens=20)
        else:
            raise ValueError("Please specify a valid pipeline or model!")
        
        re_gen_texts = [re_generated['generated_text'] for re_generated in re_generateds]
        re_gen_confs = [self.vb.compute_scores(prompt, re_gen_text) for re_gen_text in re_gen_texts]

        matches = [self.score.compute(references=[gen_text], predictions=[re_gen_text])[self.score_name] for re_gen_text in re_gen_texts]
        scores = [match*np.abs(gen_conf+re_gen_conf)+(1-match)*np.abs(1- re_gen_conf) for match, re_gen_conf in zip(matches, re_gen_confs)]
        return np.mean(scores)
    