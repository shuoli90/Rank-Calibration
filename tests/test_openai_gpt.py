import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from models import gpt

if __name__ == '__main__':
    prompts = ['Question: '+"Who is the 46th president of the United States?" + ' Answer:']
    model = gpt.GPTModel()
    responses = model.generate(prompts, max_tokens=50, n=1)
    print(responses)