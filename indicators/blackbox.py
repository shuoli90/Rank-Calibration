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

def demo_perturb(demos):
    demo_list = list(permutations(demos))
    return [[*demo] for demo in demo_list]

def spectral_projected(eigv_threshold, affinity_mode, temperature, sim_mats):
    # sim_mats: list of similarity matrices using semantic similarity model or jacard similarity
    clusterer = pc.SpetralClustering(affinity_mode=affinity_mode, eigv_threshold=eigv_threshold,
                                                cluster=False, temperature=temperature)
    return [clusterer.proj(_) for _ in sim_mats]

def jaccard_similarity(prompts, generations):
    # accept generations for batched queries
    rets = []
    for gen in generations:
        all_answers = [set(ans.lower().split()) for ans in gen]
        ret = np.eye(len(all_answers))
        for i, ans_i in enumerate(all_answers):
            for j, ans_j in enumerate(all_answers[i+1:], i+1):
                ret[i,j] = ret[j,i] = len(ans_i.intersection(ans_j)) / max(len(ans_i.union(ans_j)),1)
        rets.append(ret)
    return rets

class SemanticConsistency(BlackBox):
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

class ICLRobust(BlackBox):
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
    
    def compute_scores(self, correctness_measure, demonstrations, prompt, **kwargs):
        generations = self.generate(demonstrations, prompt, **kwargs)
        return correctness_measure(prompt, generations)

class ReparaphraseRobust(BlackBox):

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
    def __init__(self, eigv_threshold, affinity_mode, temperature):
        self.eigv_threshold = eigv_threshold
        self.affinity_mode = affinity_mode
        self.temperature = temperature
    
    def compute_scores(self, sim_mats):
        projected = spectral_projected(self.eigv_threshold, self.affinity_mode, self.temperature, sim_mats)
        ds = np.asarray([np.linalg.norm(x -x.mean(0)[None, :],2,axis=1) for x in projected])
        return np.linalg.norm(ds, 2,1), ds
    
class Degree(BlackBox):
    def __init__(self, affinity_mode, temperature):
        self.affinity_mode = affinity_mode
        self.temperature = temperature
    
    def compute_scores(self, sim_mats):
        Ws = [pc.get_affinity_mat(_, self.affinity_mode, self.temperature, symmetric=False) for _ in sim_mats]
        ret = np.asarray([np.sum(1-_, axis=1) for _ in Ws])
        return ret.mean(1), ret
        
class SpectralEigv(BlackBox):
    def __init__(self, affinity_mode, temperature, adjust):
        self.affinity_mode = affinity_mode
        self.temperature = temperature
        self.adjust = adjust
        if affinity_mode == 'jaccard':
            self.consistency = jaccard_similarity
        else:
            nlimodel = opensource.NLIModel(device='cuda')
            self.consistency = SemanticConsistency(nlimodel).similarity_mat

    def compute_scores(self, prompt, gen_text, **kwargs):
        sim_mats = self.consistency(prompt, gen_text)
        clusterer = pc.SpetralClustering(affinity_mode=self.affinity_mode, eigv_threshold=None,
                                                   cluster=False, temperature=self.temperature)
        return [clusterer.get_eigvs(_).clip(0 if self.adjust else -1).sum() for _ in sim_mats]

class SelfConsistencyConfidence(BlackBox):
    def __init__(self, pipe, score_name='exact_match'):
        self.pipe = pipe
        self.score_name = score_name
        self.score = evaluate.load(score_name)

    def compute_scores(self, prompt, gen_text, num_add_trials=5, **kwargs):
        # for a single (query, gen_text) pair
        re_generateds = self.pipe.generate(prompt, num_return_sequences=num_add_trials, max_length=50, do_sample=True, return_full_text=False)
        re_gen_texts = [text_processing.clean_generation(re_generated['generated_text']) for re_generated in re_generateds]
        scores = self.score.compute(references=[gen_text]*num_add_trials, predictions=re_gen_texts)[self.score_name]
        return np.mean(scores)

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
            raise ValueError("No formatted verbalized confidence available!")

    def compute_scores(self, prompt, gen_text, **kwargs):
        # for a single (query, gen_text) pair
        combo_text = self.description1+prompt+gen_text+self.description2
        cur_length = len(self.tokenizer(combo_text)['input_ids'])
        if self.pipe:
            verbal_conf = self.pipe.generate(combo_text, max_length=cur_length+10, return_full_text=False)[0]['generated_text']
        else:
            raise ValueError("Please specify a valid pipeline or model!")
        return self.extract_confidence(verbal_conf)

class HybridConfidence(BlackBox):
    # https://arxiv.org/pdf/2306.13063.pdf
    # use self consistency emsemble with verbalized confidences
    def __init__(self, pipe=None, score_name='exact_match'):
        self.pipe = pipe
        self.score_name = score_name
        self.score = evaluate.load(score_name)
        self.vb = VerbalizedConfidence(pipe=pipe)

    def compute_scores(self, prompt, gen_text, num_add_trials=5, **kwargs):
        # for a single (query, gen_text) pair
        gen_conf = self.vb.compute_scores(prompt, gen_text)
        if self.pipe:
            re_generateds = self.pipe.generate(prompt, num_return_sequences=num_add_trials, max_length=50, do_sample=True, return_full_text=False)
        else:
            raise ValueError("Please specify a valid pipeline")
        
        re_gen_texts = [re_generated['generated_text'] for re_generated in re_generateds]
        re_gen_confs = [self.vb.compute_scores(prompt, re_gen_text) for re_gen_text in re_gen_texts]

        matches = [self.score.compute(references=[gen_text], predictions=[re_gen_text])[self.score_name] for re_gen_text in re_gen_texts]
        scores = [match*np.abs(gen_conf+re_gen_conf)+(1-match)*np.abs(1- re_gen_conf) for match, re_gen_conf in zip(matches, re_gen_confs)]
        return np.mean(scores)
    