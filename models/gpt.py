import os
from pathlib import Path
from dotenv import load_dotenv
import openai
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
    api_key=os.getenv('OPENAI_API_KEY'),
)

def chat_gpt(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def chatcompletions_with_backoff(model, messages, n, **kwargs):
    return client.chat.completions.create(
        model=model, 
        messages=messages,
        n=n,)
class GPTModel:

    def __init__(self, model_name="gpt-3.5-turbo-0613", num_return_sequences=1, **kwargs):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained("openai-gpt")
        self.num_return_sequences = num_return_sequences

    def generate(self, prompts, **kwargs):

        messages = [{"role": "user",
                    "content": prompt} for prompt in prompts]
        response = chatcompletions_with_backoff(
            model=self.model_name,
            messages=messages,
            n=self.num_return_sequences,
        )
        responses = [{'generated_text': choice.message.content.strip()}
                for choice
                in response.choices]
        return responses