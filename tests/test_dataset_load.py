import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from tasks import factoid
from transformers import AutoTokenizer

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('../LLM/llama2/Llama-2-7b-hf')
    nq_open = factoid.NQ_Open(tokenizer, split='validation')

    nq_open.get_dataset()
    print(nq_open.tokenizer(nq_open.dataset[0]['prompt'], padding=False, truncation=False))
    # dataset = nq_open.get_dataset()
    # print(dataset[0])