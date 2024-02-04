import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from indicators import blackbox
from models import opensource, gpt
from transformers import AutoTokenizer

if __name__ == '__main__':
    # prompt = "Once upon a time:"
    # demos = ['90%: confident', '75: probably', '10%: very unlikely']
    # demons = blackbox.demo_perturb(demos)
    # prompts = [" ".join([*demo, prompt]) for demo in demons]

    # pipe = opensource.TextGenerationModel(model_name="facebook/opt-350m", torch_dtype=torch.bfloat16)
    # iclrobust = blackbox.ICLRobust(pipe=pipe, demo_transforms=blackbox.demo_perturb)
    # generations = iclrobust.generate(demos, prompt, max_length=50)
    prompt = 'Question: '+"Who is the 40th president of the United States?" + ' Answer:'

    # pipe = opensource.TextGenerationModel(model_name="facebook/opt-350m", torch_dtype=torch.bfloat16)

    # pipe = opensource.TextGenerationModel(model_name='meta-llama/Llama-2-7b-hf', torch_dtype=torch.bfloat16)
    # generated = pipe.generate(prompt, max_length=50, do_sample=True, return_full_text=False)
    # gen_text = opensource.TextGenerationModel.clean_generation(generated[0]['generated_text'])
    # sc = blackbox.SelfConsistency(pipe=pipe)
    # output = sc.compute_scores(prompt, gen_text)


    # model = gpt.GPTModel()
    # generated = model.generate([prompt], max_tokens=50, n=1)
    # gen_text = opensource.TextGenerationModel.clean_generation(generated[0]['generated_text'])
    
    # vb = blackbox.Verbalized(model=model, tokenizer=AutoTokenizer.from_pretrained("openai-gpt"))
    # output = vb.compute_scores(prompt, gen_text)


    model = gpt.GPTModel()
    generated = model.generate([prompt], max_tokens=50, num_return_sequences=1)
    gen_text = opensource.TextGenerationModel.clean_generation(generated[0]['generated_text'])
    
    hb = blackbox.Hybrid(model=model)
    output = hb.compute_scores(prompt, gen_text)
    print(output)