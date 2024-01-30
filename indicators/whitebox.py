import torch
import transformers
import functools
from models.opensource import NLIModel
from collections import defaultdict

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

            # This computation of the negative log likelihoods follows this tutorial: https://huggingface.co/docs/transformers/perplexity
            target_ids = generation.clone()
            target_ids[:len(prompt)] = -100
            model_output = model(torch.reshape(generation, (1, -1)), labels=target_ids, output_hidden_states=True)
            generation_only = generation.clone()[(len(prompt) - 1):]
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

        most_likely_generation = sample['most_likely_generation_ids'].to(device)
        target_ids = most_likely_generation.clone()
        target_ids[:len(prompt)] = -100
        model_output = model(torch.reshape(most_likely_generation, (1, -1)),
                                labels=target_ids,
                                output_hidden_states=True)
        hidden_states = model_output['hidden_states']
        average_neg_log_likelihood_of_most_likely_gen = model_output['loss']
        most_likely_generation_embedding = torch.mean(hidden_states[-1], dim=1)

        # second_most_likely_generation = sample['second_most_likely_generation_ids'].to(device)
        # target_ids = second_most_likely_generation.clone()
        # target_ids[:len(prompt)] = -100
        # model_output = model(torch.reshape(second_most_likely_generation, (1, -1)),
        #                         labels=target_ids,
        #                         output_hidden_states=True)
        # hidden_states = model_output['hidden_states']
        # average_neg_log_likelihood_of_second_most_likely_gen = model_output['loss']
        # second_most_likely_generation_embedding = torch.mean(hidden_states[-1], dim=1)

        neg_log_likelihood_of_most_likely_gen = average_neg_log_likelihood_of_most_likely_gen * (
            len(most_likely_generation) - len(prompt))

        sequence_embeddings = torch.stack(sequence_embeddings)
        result_dict['prompt'] = prompt
        result_dict['generations'] = generations
        result_dict['average_neg_log_likelihoods'] = average_neg_log_likelihoods
        result_dict['neg_log_likelihoods'] = neg_log_likelihoods
        result_dict['sequence_embeddings'] = most_likely_generation_embedding
        result_dict['most_likely_sequence_embedding'] = most_likely_generation
        result_dict['average_unconditioned_neg_log_likelihoods'] = average_unconditioned_neg_log_likelihoods
        result_dict['neg_unconditioned_log_likelihoods'] = neg_unconditioned_log_likelihoods
        result_dict['pointwise_mutual_information'] = pointwise_mutual_information
        result_dict['average_neg_log_likelihood_of_most_likely_gen'] = average_neg_log_likelihood_of_most_likely_gen
        # result_dict[
            # 'average_neg_log_likelihood_of_second_most_likely_gen'] = average_neg_log_likelihood_of_second_most_likely_gen
        result_dict['neg_log_likelihood_of_most_likely_gen'] = neg_log_likelihood_of_most_likely_gen
        # result_dict['semantic_set_ids'] = torch.tensor(similarities_dict[id_[0]]['semantic_set_ids'], device=device)
        result_dict['id'] = id_
        result.append(result_dict)

    return result
    
class WhiteBox():

    def __init__(self):
        pass

    def similarities(self):
        return NotImplementedError

def _create_semantic_sets(sample):
    # https://github.com/lorenzkuhn/semantic_uncertainty
    generated_texts = sample['mapping']
    sim_mat = sample['sim_mat'].argmax(axis=-1)
    # unique_ans is also a list of integers.
    unique_generated_texts = sorted(list(set(generated_texts)))
    semantic_set_ids = {ans: i for i, ans in enumerate(unique_generated_texts)} # one id for each exact-match answer
    for i, ans_i in enumerate(unique_generated_texts):
        for j, ans_j in enumerate(unique_generated_texts[i+1:], i+1):
            if min(sim_mat[ans_i,ans_j], sim_mat[ans_j,ans_i]) > CONTRADICT:
                semantic_set_ids[ans_j] = semantic_set_ids[ans_i]

    list_of_semantic_set_ids = [semantic_set_ids[x] for x in generated_texts]
    # map according to the order of appearance
    _map = defaultdict(int)
    ret = []
    for i, ans in enumerate(list_of_semantic_set_ids):
        if ans not in _map:
            _map[ans] = len(_map)
        ret.append(_map[ans])
    return ret


class SemanticEntropy(WhiteBox):

    def __init__(self, device=None, generations=None):
        self.device = device if device is not None else torch.device('cpu')
        self.similarity_model = NLIModel(device=self.device)
        self.mem = defaultdict(dict)
        if generations is not None:
            self.generations = generations
        else:
            # load in generations
            return NotImplementedError

    @functools.cached_property
    def similarities(self):
        sims = [self.similarity_model.classify(g['question'], g['answers']) for g in self.generations]
        return sims
    
    def _get_semantic_ids(self):
        return [_create_semantic_sets(s) for s in self.similarities]
    
    # whitebox methods
    def get_semantic_entropy(self, num_gens:int, normalize:bool):
        if self.likelihoods is None:
            return None
        semantic_set_ids = self._get_semantic_ids(num_gens)
        nlls = self.likelihoods['generations|neg_log_likelihood'][:, :num_gens]
        if normalize:
            nlls = nlls / self.likelihoods['generations|length'][:, :num_gens]
        return _hard_semantic_entropies(nlls, torch.tensor(semantic_set_ids))


        