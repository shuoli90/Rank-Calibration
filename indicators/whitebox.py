import torch
import functools
from models.opensource import NLIModel, TextGenerationModel
from collections import defaultdict

CONTRADICT, NEUTRAL, AGREE = 0, 1, 2
llh_shift = torch.tensor(5.0)

@torch.no_grad()
def get_neg_loglikelihoods(model, tokenizer, sequences):
    # https://github.com/lorenzkuhn/semantic_uncertainty
    device = model.device
    result = []
    for sample in sequences:
        result_dict = {}
        prompt = sample['prompt']
        if 'cleaned_generations' in sample:
            generations = sample['cleaned_generations'].to(device)
        else:
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
            target_ids = generation.clone()
            model_output = model(torch.reshape(generation, (1, -1)), labels=target_ids, output_hidden_states=True)
            generation_only = generation.clone()
            unconditioned_model_output = model(torch.reshape(generation_only, (1, -1)),
                                                labels=generation_only,
                                                output_hidden_states=True)
            hidden_states = model_output['hidden_states']
            average_neg_log_likelihood = model_output['loss']

            average_unconditioned_neg_log_likelihood = unconditioned_model_output['loss']
            average_neg_log_likelihoods[generation_index] = average_neg_log_likelihood
            average_unconditioned_neg_log_likelihoods[generation_index] = average_unconditioned_neg_log_likelihood
            neg_log_likelihoods[generation_index] = average_neg_log_likelihood * (len(generation) - len(prompt))
            neg_unconditioned_log_likelihoods[generation_index] = average_unconditioned_neg_log_likelihood * (
                len(generation) - len(prompt))
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

# def _logmeanexp(x, dim, ignore_negative_inf=False):
#     if ignore_negative_inf:
#         cnt = (x > -torch.inf).sum(dim)
#     else:
#         cnt = torch.tensor(x.shape[dim])
#     return torch.logsumexp(x, dim=dim) - torch.log(cnt)
#     entropy = - torch.sum(aggregated_likelihoods, dim=0) / torch.tensor(aggregated_likelihoods.shape[0])
#         entropies.append(entropy)
#     return 
    
class WhiteBox():

    def __init__(self):
        pass

    def similarities(self):
        return NotImplementedError


class SemanticEntropy(WhiteBox):

    def __init__(self, prompts, generateds, model, tokenizer, device='cuda'):
        self.device = device if device is not None else torch.device('cpu')
        self.similarity_model = NLIModel(device=self.device)
        self.mem = defaultdict(dict)
        self.model = model
        self.tokenizer = tokenizer
        gen_texts = [[TextGenerationModel.clean_generation(gen['generated_text']) for gen in generated] for generated in generateds]
        self.sequences = [{
            'prompt': torch.tensor(tokenizer.encode(prompt)).to(self.device),
            'generations': torch.tensor(tokenizer(gen_text, padding='longest')['input_ids']).to(self.device),
            'id': 0
        } for prompt, gen_text in zip(prompts, gen_texts)]
        self.generations = [{'question':prompt, 'answers':gen_text} for prompt, gen_text in zip(prompts, gen_texts)]

    @functools.cached_property
    def similarities(self):
        sims = [self.similarity_model.classify(g['question'], g['answers']) for g in self.generations]
        return sims
    
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
    
    @functools.cached_property
    def neg_log_likelihoods(self):
        return get_neg_loglikelihoods(self.model, self.tokenizer, self.sequences)
    
    @functools.cached_property
    def semantic_ids(self):
        return [self._create_semantic_sets(s) for s in self.similarities]
    
    def compute_scores(self, normalize=True):
        # https://github.com/lorenzkuhn/semantic_uncertainty
        # https://github.com/zlin7/UQ-NLG
        log_likelihoods = -torch.stack([s['average_neg_log_likelihoods'] for s in self.neg_log_likelihoods]) if normalize else -torch.stack([s['neg_log_likelihoods'] for s in self.neg_log_likelihoods])
        log_likelihoods = log_likelihoods.to(self.device)
        semantic_set_ids = torch.tensor(self.semantic_ids, device=self.device)
        num_samples = log_likelihoods.shape[0]

        entropies = []
        for num_sample in range(num_samples):
            max_num_semantic_ids = semantic_set_ids[num_sample].max().item() + 1
            aggregated_likelihoods = torch.log(torch.zeros((max_num_semantic_ids,)))
            for semantic_set_id in torch.unique(semantic_set_ids[num_sample]):
                temp = torch.where(semantic_set_ids[num_sample] == semantic_set_id, log_likelihoods[num_sample], -torch.inf)
                aggregated_likelihoods[semantic_set_id] = torch.logsumexp(temp, 0)
            aggregated_likelihoods = aggregated_likelihoods - llh_shift
            entropy = - torch.sum(aggregated_likelihoods) / torch.tensor(aggregated_likelihoods.shape[0])
            entropies.append(entropy)
        return entropies

        