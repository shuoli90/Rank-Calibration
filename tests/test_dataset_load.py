import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from tasks import closedbook, openbook, multichoice, longform
from transformers import AutoTokenizer

if __name__ == '__main__':
    Meadow = longform.Meadow()
    dataset = Meadow.get_dataset()