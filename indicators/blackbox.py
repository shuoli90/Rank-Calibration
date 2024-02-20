import torch
import functools
import evaluate
import numpy as np
from collections import defaultdict
from itertools import permutations
import re
import numpy as np
import utils.clustering as pc
from utils import text_processing 
from models import opensource

CONTRADICT, NEUTRAL, AGREE = 0, 1, 2

class BlackBox():

    def __init__(self):
        return NotImplementedError

    def compute_scores(self):
        return NotImplementedError

def demo_perturb(demonstrations):
    '''
    Input:
        demonstrations: demonstrations [d_1, ..., d_k]
    Output:
        all possible permutated ordering of demonstrations [[d_{i_1}, ..., d_{i_k}] * k!]
    '''
    demo_list = list(permutations(demonstrations))
    return [[*demo] for demo in demo_list]

def spectral_projected(affinity_mode, batch_sim_mat, threshold=0.1):
    # sim_mats: list of similarity matrices using semantic similarity model or jacard similarity
    clusterer = pc.SpetralClustering(affinity_mode=affinity_mode, cluster=False, eigv_threshold=threshold)
    return [clusterer.proj(sim_mat) for sim_mat in batch_sim_mat]

def jaccard_similarity(batch_sequences):
    '''
    Input:
        batch_sequences: a batch of sequences [[s_1^1, ..., s_{n_1}^1], ..., [s_1^1, ..., s_{n_B}^B]]
    Output:
        batch_sim_mat: a batch of real-valued similairty matrices [S^1, ..., S^B]
    '''
    batch_sim_mats = []
    for sequences in batch_sequences:
        wordy_sets = [set(seq.lower().split()) for seq in sequences]
        mat = np.eye(len(wordy_sets))
        for i, set_i in enumerate(wordy_sets):
            for j, set_j in enumerate(wordy_sets[i+1:], i+1):
                mat[i,j] = mat[j,i] = len(set_i.intersection(set_j)) / max(len(set_i.union(set_j)),1)
        batch_sim_mats.append(mat)
    return batch_sim_mats

class SemanticConsistency(BlackBox):
    def __init__(self, similarity_model=None, device='cuda'):
        self.device = device if device is not None else torch.device('cpu')
        if not similarity_model:
            self.similarity_model = opensource.NLIModel(device=device)
        else:
            self.similarity_model = similarity_model
    
    def similarity_mat(self, prompts, sequences):
        '''
        Input:
            prompts: a batch of prompt [p^1, ..., p^B]
            sequences: a batch of sequences [[s_1^1, ..., s_{n_1}^1], ..., [s_1^1, ..., s_{n_B}^B]]
        Output:
            batch_sim_mat: a batch of real-valued similairty matrices [S^1, ..., S^B]
        '''
        sims = [self.similarity_model.classify(prompts, seq) for seq in sequences]
        return [s['sim_mat'] for s in sims]

class ICLRobust(BlackBox):
    # to be revised
    def __init__(self, pipe, demo_transforms=None):
        self.pipe = pipe
        self.demo_transforms = demo_transforms
    
    def generate(self, prompt, demonstrations, **kwargs):
        demonstrations_list = self.demo_transforms(demonstrations) if self.demo_transforms else [demonstrations]
        prompts = [" ".join([*demo, prompt]) for demo in demonstrations_list]
        generations = [self.pipe.generate(prompt_tmp, **kwargs) for prompt_tmp in prompts]
        return generations
    
    def compute_scores(self, batch_prompt, demonstrations, correctness_measure, **kwargs):
        '''
        Input:
            batch_prompt: a batch of prompt [p^1, ..., p^B]
        Output:
           # to be revised# to be revised# to be revised# to be revised
        '''
        generations = [self.generate(prompt, demonstrations, **kwargs) for prompt in batch_prompt]
        return [correctness_measure(prompt, generations) for prompt in batch_prompt]

class ReparaphraseRobust(BlackBox):
    # to be unified
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

    def compute_scores(self, correctness_measure, prompt, **kwargs):
        raise NotImplementedError
    
class Eccentricity(BlackBox):
    def __init__(self, affinity_mode='disagreement', semantic_model=None, device='cuda'):
        self.affinity_mode = affinity_mode
        if affinity_mode != 'jaccard' and not semantic_model:
            self.sm = SemanticConsistency(opensource.NLIModel(device=device))
    
    def compute_scores(self, batch_prompts, batch_responses, **kwargs):
        '''
        Input:
            batch_prompts: a batch of prompts[prompt_1, ..., prompt_B]
            batch_responses: a batch of sequences [[r_1^1, ..., r_{n_1}^1], ..., [r_1^1, ..., r_{n_B}^B]]
        Output:
            batch_U: a batch of uncertainties [U^1, ..., U^B]
            batch_Cs: a batch of confidence sequences [[C_1^1, ..., C_{n_1}^1], ..., [C_1^B, ..., C_{n_B}^B]]
        '''
        batch_sim_mats = jaccard_similarity(batch_responses) if self.affinity_mode == 'jaccard' else self.sm.similarity_mat(batch_prompts, batch_responses)
        batch_projected = spectral_projected(self.affinity_mode, batch_sim_mats, threshold=0.1)
        batch_Cs = [-np.linalg.norm(projected-projected.mean(0)[None, :],2,axis=1) for projected in batch_projected]
        batch_U = [np.linalg.norm(projected-projected.mean(0)[None, :],2).clip(-1, 1) for projected in batch_projected]
        return batch_U, batch_Cs
    
class Degree(BlackBox):
    def __init__(self, affinity_mode='disagreement', semantic_model=None, device='cuda'):
        self.affinity_mode = affinity_mode
        if affinity_mode != 'jaccard' and not semantic_model:
            self.sm = SemanticConsistency(opensource.NLIModel(device=device))
    
    def compute_scores(self, batch_prompts, batch_responses, **kwargs):
        '''
        Input:
            batch_prompts: a batch of prompts [p^1, ..., p^B]
            batch_responses: a batch of sequences [[r_1^1, ..., r_{n_1}^1], ..., [r_1^1, ..., r_{n_B}^B]]
        Output:
            batch_U: a batch of uncertainties [U^1, ..., U^B]
            batch_Cs: a batch of confidence sequences [[C_1^1, ..., C_{n_1}^1], ..., [C_1^B, ..., C_{n_B}^B]]
        '''
        batch_sim_mat = jaccard_similarity(batch_responses) if self.affinity_mode == 'jaccard' else self.sm.similarity_mat(batch_prompts, batch_responses)
        batch_W = [pc.get_affinity_mat(sim_mat, self.affinity_mode) for sim_mat in batch_sim_mat]
        batch_Cs = [np.mean(W, axis=1) for W in batch_W]
        batch_U = [1/W.shape[0]-np.sum(W)/W.shape[0]**2 for W in batch_W]
        return batch_U, batch_Cs

class SpectralEigv(BlackBox):
    def __init__(self, affinity_mode, temperature=1.0, semantic_model=None, adjust=False, device='cuda'):
        self.affinity_mode = affinity_mode
        self.temperature = temperature
        self.adjust = adjust
        if affinity_mode == 'jaccard':
            self.consistency = jaccard_similarity
        else:
            nlimodel = opensource.NLIModel(device='cuda')
            self.sm = SemanticConsistency(nlimodel)

    def compute_scores(self, batch_prompts, batch_responses, **kwargs):
        sim_mats = jaccard_similarity(batch_responses) if self.affinity_mode == 'jaccard' else self.sm.similarity_mat(batch_prompts, batch_responses)
        clusterer = pc.SpetralClustering(affinity_mode=self.affinity_mode, eigv_threshold=None,
                                                   cluster=False, temperature=self.temperature)
        return [clusterer.get_eigvs(_).clip(0 if self.adjust else -1).sum() for _ in sim_mats]

class SelfConsistencyConfidence(BlackBox):
    def __init__(self, pipe, score_name='exact_match'):
        self.pipe = pipe
        self.score_name = score_name
        self.score = evaluate.load(score_name)

    def compute_scores(self, batch_prompt, batch_responses, num_add_trials=5, **kwargs):
        '''
        Input:
            batch_prompt: a batch of prompt [p^1, ..., p^B]
            batch_responses: a batch of sequences [[r_1^1, ..., r_{n_1}^1], ..., [r_1^1, ..., r_{n_B}^B]]
        Output:
            batch_Cs: a batch of confidence sequences [[C_1^1, ..., C_{n_1}^1], ..., [C_1^B, ..., C_{n_B}^B]]
        '''
        batch_Cs= []
        for prompt, responses in zip(batch_prompt, batch_responses):
            Cs = []
            for resp in responses:
                re_generateds = self.pipe.generate(prompt, num_return_sequences=num_add_trials, max_length=50, do_sample=True, return_full_text=False)
                re_gen_texts = [text_processing.clean_generation(re_generated['generated_text']) for re_generated in re_generateds]
                consist_confs = [self.score.compute(references=[resp], predictions=[re_gen_text])[self.score_name] for re_gen_text in re_gen_texts]
                Cs.append(np.mean(consist_confs))
            batch_Cs.append(Cs)
        return batch_Cs

class VerbalizedConfidence(BlackBox):
    def __init__(self, pipe=None):
        self.pipe = pipe
        self.tokenizer = self.pipe.tokenizer
        self.description1 = "Read the question and answer.\n"
        self.description2 = "\nProvide a numeric confidence that indicates your certainty about this answer. \
                            For instance, if your confidence level is 80%, it means you are 80% certain that this answer is correct and there is a 20% chance that it is incorrect. \
                            Use the following format to provide your confidence: Confidence: [Your confidence, a numerical number in the range of 0-100]%."

    def extract_confidence(self, s):
        pattern = r'Confidence:\s*(\d+)%'
        match = re.findall(pattern, s)
        if match:
            conf = int(match[0])/100
            return conf
        else:
            breakpoint()
            raise ValueError("No formatted verbalized confidence available!")

    def compute_scores(self, batch_prompt, batch_responses, **kwargs):
        '''
        Input:
            batch_prompt: a batch of prompt [p^1, ..., p^B]
            batch_responses: a batch of sequences [[r_1^1, ..., r_{n_1}^1], ..., [r_1^1, ..., r_{n_B}^B]]
        Output:
            batch_Cs: a batch of confidence sequences [[C_1^1, ..., C_{n_1}^1], ..., [C_1^B, ..., C_{n_B}^B]]
        '''
        batch_Cs = []
        for prompt, responses in zip(batch_prompt, batch_responses):
            Cs = []
            for resp in responses:
                combo_text = self.description1+prompt+resp+self.description2
                cur_length = len(self.tokenizer(combo_text)['input_ids'])
                verbal_conf = self.pipe.generate(combo_text, max_length=cur_length+10, return_full_text=False)[0]['generated_text']
                Cs.append(self.extract_confidence(verbal_conf))
            batch_Cs.append(Cs)
        return batch_Cs

class HybridConfidence(BlackBox):
    # https://arxiv.org/pdf/2306.13063.pdf
    # use self consistency emsemble with verbalized confidences
    def __init__(self, pipe=None, score_name='exact_match'):
        self.pipe = pipe
        self.score_name = score_name
        self.score = evaluate.load(score_name)
        self.vb = VerbalizedConfidence(pipe=pipe)

    def compute_scores(self, batch_prompt, batch_responses, num_add_trials=5, **kwargs):
        '''
        Input:
            batch_prompt: a batch of prompt [p^1, ..., p^B]
            batch_responses: a batch of sequences [[r_1^1, ..., r_{n_1}^1], ..., [r_1^1, ..., r_{n_B}^B]]
        Output:
            batch_Cs: a batch of confidence sequences [[C_1^1, ..., C_{n_1}^1], ..., [C_1^B, ..., C_{n_B}^B]]
        '''
        batch_Cs = []
        init_confs = self.vb.compute_scores(batch_prompt, batch_responses)
        for batch_idx, (prompt, responses) in enumerate(zip(batch_prompt, batch_responses)):
            Cs = []
            for resp_idx, resp in enumerate(responses):
                re_generateds = self.pipe.generate(prompt, num_return_sequences=num_add_trials, max_length=50, do_sample=True, return_full_text=False)
                re_gen_texts = [opensource.TextGenerationModel.clean_generation(re_generated['generated_text']) for re_generated in re_generateds]
                re_gen_confs = self.vb.compute_scores([prompt], [re_gen_texts])
                matches = [self.score.compute(references=[resp], predictions=[re_gen_text])[self.score_name] for re_gen_text in re_gen_texts]
                consist_confs = [match*np.abs((init_confs[batch_idx][resp_idx]+re_gen_conf)/2)+(1-match)*np.abs(1-re_gen_conf) for match, re_gen_conf in zip(matches, re_gen_confs[0])]
                Cs.append(np.mean(consist_confs))
            batch_Cs.append(Cs)
        return batch_Cs
    
