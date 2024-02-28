import torch
import functools
from collections import defaultdict
from evaluate import load
from models.opensource import NLIModel, TextGenerationModel

CONTRADICT, NEUTRAL, AGREE = 0, 1, 2
llh_shift = torch.tensor(5.0)

@torch.no_grad()
def get_neg_loglikelihoods(model, tokenizer, messages):
    # https://github.com/lorenzkuhn/semantic_uncertainty
    device = model.device
    result = []
    for sample in messages:
        result_dict = {}
        prompt = sample['prompt']
        generations = sample['generations'].to(device)
        id_ = sample['id']

        average_neg_log_likelihoods = torch.zeros((generations.shape[0],))
        average_unconditioned_neg_log_likelihoods = torch.zeros((generations.shape[0],))
        neg_log_likelihoods = torch.zeros((generations.shape[0],))
        neg_unconditioned_log_likelihoods = torch.zeros((generations.shape[0],))
        pointwise_mutual_information = torch.zeros((generations.shape[0],))
        sequence_embeddings = []
        for generation_index in range(generations.shape[0]):
            prompt = prompt[prompt != tokenizer.pad_token_id]
            generation = generations[generation_index][generations[generation_index] != tokenizer.pad_token_id]
            generation_only = generation.clone()
            unconditioned_model_output = model(torch.reshape(generation_only, (1, -1)),
                                                labels=generation_only,
                                                output_hidden_states=True)

            # concatenate the prompt and the generation tokens
            generation = torch.cat((prompt, generation[1:]))
            target_ids = generation.clone()
            target_ids[:len(prompt)] = -100
            model_output = model(torch.reshape(generation, (1, -1)), labels=target_ids, output_hidden_states=True)  
            hidden_states = model_output['hidden_states']
            average_neg_log_likelihood = model_output['loss']
            average_unconditioned_neg_log_likelihood = unconditioned_model_output['loss']
            average_neg_log_likelihoods[generation_index] = average_neg_log_likelihood
            average_unconditioned_neg_log_likelihoods[generation_index] = average_unconditioned_neg_log_likelihood
            # neg_log_likelihoods[generation_index] = average_neg_log_likelihood * (len(generation) - len(prompt))
            neg_log_likelihoods[generation_index] = average_neg_log_likelihood * len(generation_only)
            neg_unconditioned_log_likelihoods[generation_index] = average_unconditioned_neg_log_likelihood * len(generation_only)
            pointwise_mutual_information[generation_index] = -neg_log_likelihoods[
                generation_index] + neg_unconditioned_log_likelihoods[generation_index]

            average_of_last_layer_token_embeddings = torch.mean(hidden_states[-1], dim=1)
            sequence_embeddings.append(average_of_last_layer_token_embeddings)

        sequence_embeddings = torch.stack(sequence_embeddings)
        result_dict['prompt'] = prompt
        result_dict['generations'] = generations
        result_dict['average_neg_log_likelihoods'] = average_neg_log_likelihoods
        result_dict['neg_log_likelihoods'] = neg_log_likelihoods
        result_dict['average_unconditioned_neg_log_likelihoods'] = average_unconditioned_neg_log_likelihoods
        result_dict['neg_unconditioned_log_likelihoods'] = neg_unconditioned_log_likelihoods
        result_dict['pointwise_mutual_information'] = pointwise_mutual_information
        result_dict['id'] = id_
        result.append(result_dict)

    return result

# whitebox methods
def _logmeanexp(x, dim, ignore_negative_inf=False):
    if ignore_negative_inf:
        cnt = (x > -torch.inf).sum(dim)
    else:
        cnt = torch.tensor(x.shape[dim])
    return torch.logsumexp(x, dim=dim) - torch.log(cnt)
    
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
    
    def compute_scores(self, batch_prompt, batch_responses, nlls=None):
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

class GenerationProbability(WhiteBox):
    def __init__(self, pipe):
        self.model = pipe.model
        self.tokenizer = pipe.tokenizer
        self.device = self.model.device
    
    def compute_scores(self, batch_prompts, batch_responses):
        '''
        Input:
            batch_prompt: a batch of prompt [p^1, ..., p^B]
            batch_responses: a batch of sequences [[r_1^1, ..., r_{n_1}^1], ..., [r_1^1, ..., r_{n_B}^B]]
        Output:
            batch_GPs: a batch of probabilistic information [dict^1, ..., dict^B] where each dict contains 8 items
        '''
        messages = [{
            'prompt': torch.tensor(self.tokenizer.encode(prompt)).to(self.device),
            'generations': torch.tensor(self.tokenizer(sequences, padding='longest')['input_ids']).to(self.device),
            'id': 0
        } for prompt, sequences in zip(batch_prompts, batch_responses)]
        batch_GPs = get_neg_loglikelihoods(self.model, self.tokenizer, messages)
        return batch_GPs
    
