import evaluate
import re
from models import gpt
from models.opensource import BERTEmbedding

class Score():

    def __init__(self, metric_name='rouge', mode='rouge1'):
        self.metric_name = metric_name
        self.metric = evaluate.load(metric_name)
        self.mode = mode
    
    def __call__(self, predictions, references, use_aggregator=True):
        '''
        Input:
            prediction: a prediction [p^1, ..., p^B]
            list_references: refereces [[r_1^1, ..., r_{n_1}^1], ..., [r_1^B, ..., r_{n_B}^B]]
        Output:
            
        '''
        if 'rouge' in self.metric_name:
            score = self.metric.compute(predictions=predictions, references=references, use_aggregator=use_aggregator)
        else:
            score = self.metric.compute(predictions=predictions, references=references)
        if isinstance(score, dict):
            score = score[self.mode]
        return score

class BertSimilarity():
    def __init__(self, model_name='sentence-transformers/bert-base-nli-mean-tokens'):
        self.model = BERTEmbedding(model_name)
    
    def __call__(self, prompt, references, predictions):
        score = self.model.compare(prompt, references, predictions)
        return score

def extract_numbers(string):
    # Regular expression pattern to match numbers from 0 to 100
    pattern = r'\b(0|[1-9][0-9]?|100)\b'
    
    # Find all matches of the pattern in the string
    matches = re.findall(pattern, string)
    
    # Convert matched strings to integers
    numbers = [int(match) for match in matches]
    try:
        num = numbers[0]
        return num
    except:
        return 0.0

class ChatgptCorrectness():
    def __init__(self):
        self.model = gpt.GPTModel()
    
    def __call__(self, prompt, reference, generateds):
        scores_ret = []
        for generated in generateds:
            prompt = f"""Rate the level of consistency between the answer to the question and the reference answer, from 0 to 100.
    Question: In Scotland a bothy/bothie is a?
    Reference: House
    Answer: House
    Rating: 100.
    Question: Where in England was Dame Judi Dench born?
    Reference: York
    Answer: London
    Rating: 0.
    Question: {prompt}
    Reference: {reference}
    Answer: {generated}
    Rating:"""
            score = self.model.generate(prompt)[0][0]
            num = float(extract_numbers(score[0]))
            scores_ret.append(num)
        return scores_ret