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

dotenv_path = Path('.env')
load_dotenv(dotenv_path=dotenv_path)

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=os.getenv('OPENAI_API_KEY'),
)

def chat_gpt(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# Set up chatgpt api
# def setup_openai():
#     dotenv_path = Path('.env')
#     load_dotenv(dotenv_path=dotenv_path)
#     openai.api_key = os.getenv('OPENAI_API_KEY')
#     openai.organization = 'org-ZZanmSElxExwouVgEWnZL8zv'

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def chatcompletions_with_backoff(model, messages, **kwargs):
    return client.chat.completions.create(
        model=model, 
        messages=messages,
        **kwargs)

def generate(prompts, model="gpt-3.5-turbo-0613", **kwargs):

    messages = [{"role": "user",
                 "content": prompt} for prompt in prompts]
    response = chatcompletions_with_backoff(
        model=model,
        messages=messages,
        **kwargs,
    )

    responses = [choice.message.content.strip()
               for choice
               in response.choices]
    # input_token_counts = response.usage['prompt_tokens']
    # output_token_counts = response.usage['completion_tokens']

    return responses