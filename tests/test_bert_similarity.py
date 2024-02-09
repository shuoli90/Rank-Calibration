import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from metrics import correctness

if __name__ == '__main__':
    prompt = ['who is the president of the united states?']
    ans_1 = ['Joe Biden', 'Donald Trump']
    ans_2 = ['Donald Trump', 'Joe Biden']
    score = correctness.BertSimilarity()
    bert_score = score(prompt, ans_1, ans_2)

    score = correctness.ChatgptSimilarity()
    gpt_score = score(prompt, ans_1[0], ans_2[0])