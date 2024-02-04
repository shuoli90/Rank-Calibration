import torch
import functools
from collections import defaultdict
from evaluate import load
from models.opensource import NLIModel, TextGenerationModel

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
            neg_log_likelihoods[generation_index] = average_neg_log_likelihood * len(generation)
            neg_unconditioned_log_likelihoods[generation_index] = average_unconditioned_neg_log_likelihood * len(generation)
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

    def __init__(self, model, tokenizer, device='cuda'):
        self.device = device if device is not None else torch.device('cpu')
        self.similarity_model = NLIModel(device=self.device)
        self.mem = defaultdict(dict)
        self.model = model
        self.tokenizer = tokenizer

    def similarities(self, generations):
        sims = [self.similarity_model.classify(g['question'], g['answers']) for g in generations]
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
    
    # @functools.cached_property
    def neg_log_likelihoods(self, sequences):
        return get_neg_loglikelihoods(self.model, self.tokenizer, sequences)
    
    # @functools.cached_property
    def semantic_ids(self, generations):
        return torch.tensor([self._create_semantic_sets(s) for s in self.similarities(generations)]).to(self.device)
    
    def compute_scores(self, prompts, generateds, normalize=True):
        # https://github.com/lorenzkuhn/semantic_uncertainty
        gen_texts = [[TextGenerationModel.clean_generation(gen['generated_text']) for gen in generated] for generated in generateds]
        sequences = [{
            'prompt': torch.tensor(self.tokenizer.encode(prompt)).to(self.device),
            'generations': torch.tensor(self.tokenizer(gen_text, padding='longest')['input_ids']).to(self.device),
            'id': 0
        } for prompt, gen_text in zip(prompts, gen_texts)]
        generations = [{'question':prompt, 'answers':gen_text} for prompt, gen_text in zip(prompts, gen_texts)]

        neg_log_likelihoods = self.neg_log_likelihoods(sequences)
        semantic_set_ids = self.semantic_ids(generations)

        log_likelihoods = -torch.stack([s['average_neg_log_likelihoods'] for s in neg_log_likelihoods]) if normalize else -torch.stack([s['neg_log_likelihoods'] for s in neg_log_likelihoods])
        log_likelihoods = log_likelihoods.to(self.device)
        num_samples = log_likelihoods.shape[0]
        entropies = []
        for num_sample in range(num_samples):
            semantic_set_ids_tmp = semantic_set_ids[num_sample][~torch.isnan(log_likelihoods[num_sample])]
            log_likelihoods_tmp = log_likelihoods[num_sample][~torch.isnan(log_likelihoods[num_sample])]
            aggregated_log_likelihoods = []
            for semantic_set_id in torch.unique(semantic_set_ids_tmp):
                temp = log_likelihoods_tmp[semantic_set_ids_tmp == semantic_set_id]
                aggregated_log_likelihoods.append(torch.logsumexp(temp, 0))
            aggregated_log_likelihoods = torch.tensor(aggregated_log_likelihoods)
            entropy = - torch.sum(aggregated_log_likelihoods, dim=0) / torch.tensor(aggregated_log_likelihoods.shape[0])
            entropies.append(entropy)
        return entropies

class PerplexityScore(WhiteBox):
    def __init__(self, model, tokenizer):
        max_length = model.config.n_positions

    
    def compute_scores(self, generateds):
        gen_texts = [[TextGenerationModel.clean_generation(gen['generated_text']) for gen in generated] for generated in generateds]
        results = []
        for gen_text in gen_texts:
            result = self.perplexity.compute(predictions=gen_text, model_id=self.model)
            results.append(result)
        return results

class GenerationProbability(WhiteBox):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
    
    def compute_scores(self, prompts, generateds):
        gen_texts = [[TextGenerationModel.clean_generation(gen['generated_text']) for gen in generated] for generated in generateds]
        self.sequences = [{
            'prompt': torch.tensor(self.tokenizer.encode(prompt)).to(self.device),
            'generations': torch.tensor(self.tokenizer(gen_text, padding='longest')['input_ids']).to(self.device),
            'id': 0
        } for prompt, gen_text in zip(prompts, gen_texts)]
        results = get_neg_loglikelihoods(self.model, self.tokenizer, self.sequences)
        return results
    
