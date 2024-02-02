import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# os.environ['TRANSFORMERS_CACHE'] = '../LLM/llama2'
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import pipeline

if __name__ == '__main__':
    save_dir = '../LLM/llama2/Llama-2-7b-hf'
    pipe = pipeline(model=save_dir)
    # tokenizer = LlamaTokenizer.from_pretrained(save_dir, low_cpu_mem_usage=True)
    # model = LlamaForCausalLM.from_pretrained(save_dir, low_cpu_mem_usage=True)
