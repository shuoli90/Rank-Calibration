import datasets
import functools

@functools.lru_cache(1)
class NQ_Open:

    def __init__(self, tokenizer, split='validation'):
        self.dataset = datasets.load_dataset('nq_open', split=split)
        self.tokenizer = tokenizer
    
        def verbalize(example):
            example['prompt'] = prompt = 'Question: ' + example['question'] + " Answer: "
            return self.tokenizer(prompt, truncation=False, padding=False)

        self.dataset = self.dataset.map(verbalize, batched=False, load_from_cache_file=False)
        self.dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], output_all_columns=True)