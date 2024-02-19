import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from tasks import closedbook, openbook, multichoice
from transformers import AutoTokenizer

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
    # nq_open = closedbook.NQ_Open(tokenizer, split='validation')

    # nq_open.get_dataset()
    # print(nq_open.tokenizer(nq_open.dataset[0]['prompt'], padding=False, truncation=False))
    # dataset = nq_open.get_dataset()
    # print(dataset[0])

    # trivia = openbook.TriviaQA(split='validation').get_dataset()
    
    # squad = openbook.SQuAD(split='validation').get_dataset()
   
    # truthful = closedbook.Truthful(tokenizer, split='validation').get_dataset()

    # mmlu = multichoice.MMLU(split='validation').get_dataset(tokenizer)

    medmc = multichoice.MedMC(split='validation').get_dataset(tokenizer)