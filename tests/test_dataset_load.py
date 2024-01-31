import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from tasks import factoid
from transformers import AutoTokenizer

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
    nq_open = factoid.NQ_Open(tokenizer, split='validation')