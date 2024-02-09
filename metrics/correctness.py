import evaluate
from models import gpt
from models.opensource import BERTEmbedding

class Score():

    def __init__(self, metric_name='rouge', mode='rouge1', metric_threshold=0.5):
        self.metric = evaluate.load(metric_name)
        self.threshold = metric_threshold
        self.mode = mode
    
    def __call__(self, predictions, references):
        score = self.metric.compute(predictions=predictions, references=references)
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