from typing import Any
import datasets
import functools

@functools.lru_cache(1)
class NQ_Open:

    def __init__(self, tokenizer, split='validation'):
        self.dataset = datasets.load_dataset('nq_open', split=split)
        self.tokenizer = tokenizer
    
    def get_dataset(self, add_prompt=None):

        def process_instance(example):
            # https://github.com/zlin7/UQ-NLG
            all_answers = example.pop('answer')
            example['answer'] = all_answers
            example['prompt'] = 'Question: ' + example['question'] + ' Answer:'
            inputs = self.tokenizer(example['prompt'], padding=False, truncation=False)
            outputs = self.tokenizer(all_answers[0], padding=False, truncation=False)
            example['input_ids'] = inputs['input_ids']
            example["attention_mask"] = inputs.attention_mask
            example["labels"] = outputs.input_ids.copy()
            example["labels"] = [-100 if _ == self.tokenizer.pad_token_id else _ for _ in example["labels"]]
            return example
        
        self.dataset = self.dataset.map(process_instance, load_from_cache_file=False)
        self.dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"],
            output_all_columns=True)
        
        return self.dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

@functools.lru_cache(1)
class Truthful():

    def __init__(self, split='validation'):
        self.dataset = datasets.load_dataset('truthful_qa', 'generation', split=split)
    
    def get_dataset(self, tokenizer=None, prompt_template=None):

        def process_instance(example):
            if prompt_template is not None:
                example['prompt'] = prompt_template(example)
            else:
                question = example['question']
                answers = [example['best_answer']] + example['correct_answers']
                example['prompt'] = "Answer the following question: " + question + "[SEP]" + " Answer: "
                example['answer'] = answers
                if tokenizer is not None:
                    inputs = tokenizer(example['prompt'], padding=False, truncation=False)
                    outputs = tokenizer(answers[0], padding=False, truncation=False)
                    example['input_ids'] = inputs['input_ids']
                    example["attention_mask"] = inputs.attention_mask
                    example["labels"] = outputs.input_ids.copy()
                    example["labels"] = [-100 if _ == tokenizer.pad_token_id else _ for _ in example["labels"]]
            return example

        self.dataset = self.dataset.map(process_instance, load_from_cache_file=False)
        if tokenizer is not None:
            self.dataset.set_format(
                type="torch",
                columns=["input_ids", "attention_mask", "labels"],
                output_all_columns=True)
        return self.dataset