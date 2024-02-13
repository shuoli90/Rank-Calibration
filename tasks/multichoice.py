from typing import Any
import datasets
import functools

@functools.lru_cache(1)
class MMLU():

    def __init__(self, split='validation'):
        self.dataset = datasets.load_dataset('cais/mmlu', 'all', split=split)
    
    def get_dataset(self, tokenizer=None, prompt_template=None):

        def process_instance(example):
            if prompt_template is not None:
                example['prompt'] = prompt_template(example)
            else:
                question = example['question']
                choices = " ".join([str(ch+1) + ":" + choice for ch, choice in enumerate(example['choices'])])
                answers = str(example['answer'])
                example['prompt'] = "Answer the following question from the given choices: " + question + "[SEP]" + choices + "[SEP]" + " Answer: "
                example['answer'] = answers
                if tokenizer is not None:
                    inputs = tokenizer(example['prompt'], padding=False, truncation=False)
                    outputs = tokenizer(answers, padding=False, truncation=False)
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

@functools.lru_cache(1)
class MedMC():
    def __init__(self, split='validation'):
        self.dataset = datasets.load_dataset('medmcqa', split=split)
    
    def get_dataset(self, tokenizer=None, prompt_template=None):
        
        def process_instance(example):
            if prompt_template is not None:
                example['prompt'] = prompt_template(example)
            else:
                question = example['question']
                choices = "0:" + example['opa'] + " 1:" + example['opb'] + " 2:" + example['opc'] + " 3:" + example['opd']
                answers = str(example['cop'])
                example['prompt'] = "Answer the following question from the given choices: " + question + "[SEP]" + choices + "[SEP]" + " Answer: "
                example['answer'] = answers
                if tokenizer is not None:
                    inputs = tokenizer(example['prompt'], padding=False, truncation=False)
                    outputs = tokenizer(answers, padding=False, truncation=False)
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
    
