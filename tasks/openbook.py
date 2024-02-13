from typing import Any
import datasets
import functools

@functools.lru_cache(1)
class TriviaQA():
    def __init__(self, config='rc.wikipedia', split='validation'):
        self.split = split
        self.config = config
        self.dataset = datasets.load_dataset('trivia_qa', config, split=split)
    
    def get_dataset(self, tokenizer=None, prompt_template=None):
        
        def verbalize(example):
            if prompt_template is not None:
                example['prompt'] = prompt_template(example)
            else:
                question = example['question']
                answers = example['answer']['normalized_aliases']
                context = example['entity_pages']['wiki_context'][0]
                example['prompt'] = "Answer the following question basing on the context: " + question + "[SEP]" + " Context: " + context + "[SEP]"+ " Answer: "
                example['answer'] = answers
                if tokenizer is not None:
                    inputs = tokenizer(example['prompt'], padding=False, truncation=False)
                    outputs = tokenizer(answers[0], padding=False, truncation=False)
                    example['input_ids'] = inputs['input_ids']
                    example["attention_mask"] = inputs.attention_mask
                    example["labels"] = outputs.input_ids.copy()
                    example["labels"] = [-100 if _ == tokenizer.pad_token_id else _ for _ in example["labels"]]
            return example

        self.dataset = self.dataset.map(verbalize, load_from_cache_file=False)
        if tokenizer is not None:
            self.dataset.set_format(
                type="torch",
                columns=["input_ids", "attention_mask", "labels"],
                output_all_columns=True)
        return self.dataset

@functools.lru_cache(1)
class SQuAD():
    def __init__(self, split='validation'):
        self.split = split
        self.dataset = datasets.load_dataset('squad', split=split)
    
    def get_dataset(self, tokenizer=None, prompt_template=None):

        def verbalize(example):
            if prompt_template is not None:
                example['prompt'] = prompt_template(example)
            else:
                question = example['question']
                context = example['context']
                example['prompt'] = "Answer the following question basing on the context: " + question + " Context: " + context + " Answer: "
                answers = example['answers']['text']
                if tokenizer is not None:
                    inputs = tokenizer(example['prompt'], padding=False, truncation=False)
                    outputs = tokenizer(answers[0], padding=False, truncation=False)
                    example['input_ids'] = inputs['input_ids']
                    example["attention_mask"] = inputs.attention_mask
                    example["labels"] = outputs.input_ids.copy()
                    example["labels"] = [-100 if _ == tokenizer.pad_token_id else _ for _ in example["labels"]]
            return example

        self.dataset = self.dataset.map(verbalize, load_from_cache_file=False)
        if tokenizer is not None:
            self.dataset.set_format(
                type="torch",
                columns=["input_ids", "attention_mask", "labels"],
                output_all_columns=True)
        return self.dataset
