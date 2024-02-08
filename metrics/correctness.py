import evaluate

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
        if use_aggregator:
            return {'score': score, 'pass': score > self.threshold}
        else:
            return {'score': score, 'pass': [sc > self.threshold for sc in score]}

