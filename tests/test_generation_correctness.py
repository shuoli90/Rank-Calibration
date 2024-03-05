import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from metrics import correctness

if __name__ == '__main__':
    # metric_names = ['rouge', 'bleu', 'meteor']
    # generations = ['J. Biden', 'Morning']
    # references = [['Joe Biden', 'President Joe Biden'], ['Joe Biden', 'President Joe Biden']]
    # for metric_name in metric_names:
    #     if metric_name == 'rouge':
    #         score = correctness.Score(metric_name=metric_name, mode='rouge1')
    #     elif metric_name == 'bleu':
    #         score = correctness.Score(metric_name=metric_name, mode='bleu')
    #     elif metric_name == 'meteor':
    #         score = correctness.Score(metric_name=metric_name, mode='meteor')
    #     print(metric_name, score(generations, references))
    
    chatgpt = correctness.ChatgptCorrectness()
    prompt = 'who is the current president of the united states?'
    score = chatgpt(prompt, reference='Joe Biden', generateds=['Joe Biden', 'trump'])