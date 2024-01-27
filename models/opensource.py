import torch
from transformers import pipeline

class TextGenerationModel:
    def __init__(self, model_name='meta-llama/Llama-2-7b-hf', **kwargs):
        self.pip = pipeline(model=model_name, device_map="auto", **kwargs)

    def generate(self, prompt, **kwargs):
        return self.pip(prompt, **kwargs)