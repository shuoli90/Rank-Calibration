import torch
import functools
from itertools import permutations

def demo_perturb(demos):
    demo_list = list(permutations(demos))
    return [[*demo] for demo in demo_list]

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
        self.model = pipe
        self.prompt_transforms = prompt_transforms
    
    def generate(self, prompt, **kwargs):
        if self.prompt_transforms is not None:
            prompt_list = self.prompt_transforms(prompt)
        else:
            prompt_list = [prompt]
        generations = [self.pipe.generate(prompt, **kwargs) for prompt in prompt_list]
        return generations

    
