import torch
import functools
from collections import defaultdict
from itertools import permutations
import numpy as np
import utils.clustering as pc

CONTRADICT, NEUTRAL, AGREE = 0, 1, 2

def demo_perturb(demos):
    demo_list = list(permutations(demos))
    return [[*demo] for demo in demo_list]

def spectral_projected(eigv_threshold, affinity_mode, temperature, sim_mats):
    # sim_mats: list of similarity matrices using semantic similarity model or jacard similarity
    clusterer = pc.SpetralClusteringFromLogits(affinity_mode=affinity_mode, eigv_threshold=eigv_threshold,
                                                cluster=False, temperature=temperature)
    return [clusterer.proj(_) for _ in sim_mats]

def jaccard_similarity(generations):
    rets = []
    for gen in generations:
        all_answers = [set(ans.lower().split()) for ans in gen]
        ret = np.eye(len(all_answers))
        for i, ans_i in enumerate(all_answers):
            for j, ans_j in enumerate(all_answers[i+1:], i+1):
                ret[i,j] = ret[j,i] = len(ans_i.intersection(ans_j)) / max(len(ans_i.union(ans_j)),1)
        rets.append(ret)
    return rets

class SemanticConsistency():
    def __init__(self, similarity_model, device='cuda'):
        self.device = device if device is not None else torch.device('cpu')
        self.similarity_model = similarity_model

    def _create_semantic_sets(self, sample):
        # https://github.com/lorenzkuhn/semantic_uncertainty
        generated_texts = sample['mapping']
        sim_mat = sample['sim_mat'].argmax(axis=-1)
        unique_generated_texts = sorted(list(set(generated_texts)))
        semantic_set_ids = {ans: i for i, ans in enumerate(unique_generated_texts)} # one id for each exact-match answer
        for i, ans_i in enumerate(unique_generated_texts):
            for j, ans_j in enumerate(unique_generated_texts[i+1:], i+1):
                if min(sim_mat[ans_i,ans_j], sim_mat[ans_j,ans_i]) > CONTRADICT:
                    semantic_set_ids[ans_j] = semantic_set_ids[ans_i]

        list_of_semantic_set_ids = [semantic_set_ids[x] for x in generated_texts]
        _map = defaultdict(int)
        ret = []
        for i, ans in enumerate(list_of_semantic_set_ids):
            if ans not in _map:
                _map[ans] = len(_map)
            ret.append(_map[ans])
        return ret
    
    def __call__(self, prompt, generations):
        sims = [self.similarity_model.classify(prompt, g) for g in generations]
        clusters = torch.tensor([self._create_semantic_sets(s) for s in sims]).to(self.device)
        return clusters.max().item() + 1
    
    def similarity_mat(self, prompt, generations):
        sims = [self.similarity_model.classify(prompt, g) for g in generations]
        return [s['sim_mat'] for s in sims]

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
        return consistency_meansure(prompt, generations)

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


class Eccentricity():
    
    def __init__(self, eigv_threshold, affinity_mode, temperature):
        self.eigv_threshold = eigv_threshold
        self.affinity_mode = affinity_mode
        self.temperature = temperature
    
    def compute_scores(self, sim_mats):
        projected = spectral_projected(self.eigv_threshold, self.affinity_mode, self.temperature, sim_mats)
        ds = np.asarray([np.linalg.norm(x -x.mean(0)[None, :],2,axis=1) for x in projected])
        return np.linalg.norm(ds, 2,1), ds

class Degree():
        
        def __init__(self, affinity_mode, temperature):
            self.affinity_mode = affinity_mode
            self.temperature = temperature
        
        def compute_scores(self, sim_mats):
            Ws = [pc.get_affinity_mat(_, self.affinity_mode, self.temperature, symmetric=False) for _ in sim_mats]
            ret = np.asarray([np.sum(1-_, axis=1) for _ in Ws])
            return ret.mean(1), ret