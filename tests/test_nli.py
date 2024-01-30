import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from models import opensource

if __name__ == '__main__':
    question = 'who is the president of the united states?'
    answers = ['Donald Trump', 'Barack Obama', 'George Washington']
    nlimodel = opensource.NLIModel(device='cuda')
    results = nlimodel.classify(question, answers)
    comp = nlimodel.compare(question, answers[0], answers[1])