from typing import Any
import datasets
import functools

@functools.lru_cache(1)
class TriviaQA():
    def __init__(self, config='rc', split='validation'):
        self.split = split
        self.config = config
        data = datasets.load_dataset("trivia_qa", "rc.nocontext", split=split)
        id_mem = set()
        def remove_dups(batch):
            if batch['question_id'][0] in id_mem:
                return {_:[] for _ in batch.keys()}
            id_mem.add(batch['question_id'][0])
            return batch
        self.dataset = data.map(remove_dups, batch_size=1, batched=True, load_from_cache_file=False)
    
    def get_dataset(self, tokenizer=None, prompt_template=None):
        
        def verbalize(example):
            if prompt_template is not None:
                example['prompt'] = prompt_template(example)
            else:
                question = example['question']
                # set all answers to lowercase
                answers = [answer.lower() for answer in example['answer']['aliases']]
                context = [context.lower() for context in example['search_results']['description']]
                # if len(context) == 0:
                #     example['context'] = ""
                # else:
                #     context_true = ""
                #     for context_tmp in context:
                #         # if any answer appears in the context, we use that context
                #         if any([answer in context_tmp for answer in answers]):
                #             context_true = context_tmp
                #             break
                #     example['context'] = context_true
                # example['prompt'] = "Answer the following question shortly according to the context: " + question + "[SEP]" + " Context: " + example['context'] + "[SEP]"+ " Answer: "
                example['prompt'] = f"""
                Answer these questions:
                Q: In Scotland a bothy/bothie is a?
                A: House
                Q: {question}
                A:"""
                example['answers'] = answers
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
                example['answers'] = example['answers']['text']
                if tokenizer is not None:
                    inputs = tokenizer(example['prompt'], padding=False, truncation=False)
                    outputs = tokenizer(example['answers'][0], padding=False, truncation=False)
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
