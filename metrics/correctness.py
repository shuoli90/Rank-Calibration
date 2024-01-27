import evaluate

class Score():

    def __init__(self, metric_name='rouge', mode='rouge1', metric_threshold=0.5):
        self.score = evaluate.load(metric_name)
        self.threshold = metric_threshold
        self.mode = mode
    
    def __call__(self, predictions, references):
        score = self.score.compute(predictions=predictions, references=references)
        if isinstance(score, dict):
            score = score[self.mode]
        return {'score': score, 'pass': score > self.threshold}
