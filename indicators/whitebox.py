import torch
import functools
from collections import defaultdict
from evaluate import load
from models.opensource import NLIModel, TextGenerationModel

CONTRADICT, NEUTRAL, AGREE = 0, 1, 2
llh_shift = torch.tensor(5.0)

class WhiteBox():

    def __init__(self):
        return NotImplementedError

    def compute_scores(self):
        return NotImplementedError


class SemanticEntropy(WhiteBox):

    def __init__(self, device='cuda', similarity_model=None):
        self.device = device if device is not None else torch.device('cpu')
        if not similarity_model:
            self.similarity_model = NLIModel(device=device)
        self.mem = defaultdict(dict)

    def similarities(self, generations):
        sims = [self.similarity_model.classify(g['question'], g['answers']) for g in generations]
        return sims
    
    def _create_semantic_sets(self, sample):
        # https://github.com/lorenzkuhn/semantic_uncertainty
        generated_texts = sample['mapping']
        sim_mat = sample['sim_mat'].argmax(axis=-1)
        semantic_set_ids = {ans: i for i, ans in enumerate(generated_texts)} # one id for each exact-match answer
        for i, ans_i in enumerate(generated_texts):
            for j, ans_j in enumerate(generated_texts[i+1:], i+1):
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
    
    # @functools.cached_property
    def semantic_ids(self, batch_qa_pairs):
        return torch.tensor([self._create_semantic_sets(s) for s in self.similarities(batch_qa_pairs)]).to(self.device)
    
    def compute_scores(self, batch_prompt, batch_responses, nlls):
        '''
        Input:
            batch_prompt: a batch of prompt [p^1, ..., p^B]
            batch_responses: a batch of sequences [[r_1^1, ..., r_{n_1}^1], ..., [r_1^1, ..., r_{n_B}^B]]
        Output:
            batch_entropy: a batch of semantic entropies [etp^1, ..., etp^B]
        '''
        # https://github.com/lorenzkuhn/semantic_uncertainty
        batch_qa_pairs = [{'question':prompt, 'answers':responses} for prompt,responses in zip(batch_prompt, batch_responses)]
        semantic_set_ids = self.semantic_ids(batch_qa_pairs)

        # log_likelihoods = -torch.stack([s['average_neg_log_likelihoods'] for s in neg_log_likelihoods]) if normalize else -torch.stack([s['neg_log_likelihoods'] for s in neg_log_likelihoods])
        log_likelihoods = -torch.tensor(nlls).to(self.device)
        num_samples = log_likelihoods.shape[0]
        batch_entropy = []
        for num_sample in range(num_samples):
            semantic_set_ids_tmp = semantic_set_ids[num_sample][~torch.isnan(log_likelihoods[num_sample])]
            log_likelihoods_tmp = log_likelihoods[num_sample][~torch.isnan(log_likelihoods[num_sample])]
            aggregated_log_likelihoods = []
            for semantic_set_id in torch.unique(semantic_set_ids_tmp):
                temp = log_likelihoods_tmp[semantic_set_ids_tmp == semantic_set_id]
                aggregated_log_likelihoods.append(torch.logsumexp(temp, 0))
            aggregated_log_likelihoods = torch.tensor(aggregated_log_likelihoods)
            entropy = - torch.sum(aggregated_log_likelihoods, dim=0) / torch.tensor(aggregated_log_likelihoods.shape[0])
            batch_entropy.append(entropy)
        return batch_entropy
