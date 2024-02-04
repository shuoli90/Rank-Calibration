import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import pipeline

if __name__ == '__main__':
    pipe = pipeline(model='meta-llama/Llama-2-7b-hf')
    # tokenizer = LlamaTokenizer.from_pretrained(save_dir, low_cpu_mem_usage=True)
    # model = LlamaForCausalLM.from_pretrained(save_dir, low_cpu_mem_usage=True)
