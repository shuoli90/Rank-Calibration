import evaluate
from models import gpt
from models.opensource import BERTEmbedding

class Score():

    def __init__(self, metric_name='rouge', mode='rouge1', metric_threshold=0.5):
        self.metric = evaluate.load(metric_name)
        self.threshold = metric_threshold
        self.mode = mode
    
    def __call__(self, predictions, references, use_aggregator=True):
        '''
        Input:
            prediction: a prediction [p^1, ..., p^B]
            list_references: refereces [[r_1^1, ..., r_{n_1}^1], ..., [r_1^B, ..., r_{n_B}^B]]
        Output:
            
        '''
        score = self.metric.compute(predictions=predictions, references=references, use_aggregator=use_aggregator)
        if isinstance(score, dict):
            score = score[self.mode]
        return score

class BertSimilarity():
    def __init__(self, model_name='sentence-transformers/bert-base-nli-mean-tokens'):
        self.model = BERTEmbedding(model_name)
    
    def __call__(self, prompt, ans_1, ans_2):
        score = self.model.compare(prompt, ans_1, ans_2)
        return score

class ChatgptSimilarity():
    def __init__(self):
        self.model = gpt.GPTModel()
    
    def __call__(self, prompt, ans_1, ans_2):
        prompt = f"Return a number between 0 and 1 that evaluates the semantic similarity level between the following sentences given the prompt. prompt: {prompt} \n sentence 1: {ans_1} \n sentence 2: {ans_2}"
        scores = self.model.generate(prompt)
        scores = [float(score['generated_text']) for score in scores]
        return scores