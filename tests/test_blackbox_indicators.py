import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from indicators import blackbox
from models import opensource

if __name__ == '__main__':
    prompt = "Once upon a time:"
    demos = ['90%: confident', '75: probably', '10%: very unlikely']
    demons = blackbox.demo_perturb(demos)
    prompts = [" ".join([*demo, prompt]) for demo in demons]

    pipe = opensource.TextGenerationModel(model_name="facebook/opt-350m", torch_dtype=torch.bfloat16)
    iclrobust = blackbox.ICLRobust(pipe=pipe, demo_transforms=blackbox.demo_perturb)
    generations = iclrobust.generate(demos, prompt, max_length=50)