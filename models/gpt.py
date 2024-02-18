import os
from pathlib import Path
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
from openai import OpenAI
from transformers import AutoTokenizer

dotenv_path = Path('.env')
load_dotenv(dotenv_path=dotenv_path)

client = OpenAI(
    api_key=os.getenv('YOU_OPENAI_API_KEY'),
)

def chat_gpt(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def chatcompletions_with_backoff(model, messages, n, **kwargs):
    return client.chat.completions.create(
        model=model, 
        messages=messages,
        n=n,
        **kwargs)

class GPTModel:

    def __init__(self, model_name="gpt-3.5-turbo-0613",**kwargs):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained("openai-gpt")

    def generate(self, prompts, num_return_sequences=1, max_length=50, do_sample=True, return_full_text=False, temperature=1.0, **kwargs):

        if not isinstance(prompts, list):
            prompts = [prompts]

        messages = [{"role": "user",
                    "content": prompt} for prompt in prompts]
        completions = []
        for message in messages:
            response = chatcompletions_with_backoff(
                model=self.model_name,
                messages=[message],
                n=num_return_sequences,
                max_tokens=max_length,
                temperature=temperature if do_sample else 0,
                # **kwargs # to be refined
            )
            completions.append(response)
        responses_list = []
        for completion in completions:
            responses = [{'generated_text': choice.message.content.strip()}
                for choice
                in completion.choices]
            responses_list.append(responses)
        if return_full_text:
            for prompt, response in zip(prompts, responses_list):
                for item in response:
                    item['generated_text'] = f"{prompt} {item['generated_text']}"
        return [responses]