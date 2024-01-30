import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from models import gpt

if __name__ == '__main__':
    prompts = ['Once upon a time']
    responses = gpt.generate(prompts, model="gpt-3.5-turbo-0613", max_tokens=50, n=1)