from typing import Any
import datasets
import functools


@functools.lru_cache(1)
class Meadow():
    def __init__(self, tokenizer=None, split='train'):
        self.dataset = datasets.load_dataset('medalpaca/medical_meadow_cord19', split=split)
        self.tokenizer = tokenizer
    
    def get_dataset(self, add_prompt=None):
            
        def process_instance(example):
            example['prompt'] = f"""
            Abstract: Coronavirus disease 2019 (COVID-19) threatens vulnerable patient populations, resulting in immense pressures at the local, regional, national, and international levels to contain the virus. Laboratory-based studies demonstrate that masks may offer benefit in reducing the spread of droplet-based illnesses, but few data are available to assess mask effects via executive order on a population basis. We assess the effects of a county-wide mask order on per-population mortality, intensive care unit (ICU) utilization, and ventilator utilization in Bexar County, Texas. METHODS: We used publicly reported county-level data to perform a mixed-methods before-and-after analysis along with other sources of public data for analyses of covariance. We used a least-squares regression analysis to adjust for confounders. A Texas state-level mask order was issued on July 3, 2020, followed by a Bexar County–level order on July 15, 2020. We defined the control period as June 2 to July 2 and the postmask order period as July 8, 2020–August 12, 2020, with a 5-day gap to account for the median incubation period for cases; longer periods of 7 and 10 days were used for hospitalization and ICU admission/death, respectively. Data are reported on a per-100,000 population basis using respective US Census Bureau–reported populations. RESULTS: From June 2, 2020 through August 12, 2020, there were 40,771 reported cases of COVID-19 within Bexar County, with 470 total deaths. The average number of new cases per day within the county was 565.4 (95% confidence interval [CI] 394.6–736.2). The average number of positive hospitalized patients was 754.1 (95% CI 657.2–851.0), in the ICU was 273.1 (95% CI 238.2–308.0), and on a ventilator was 170.5 (95% CI 146.4–194.6). The average deaths per day was 6.5 (95% CI 4.4–8.6). All of the measured outcomes were higher on average in the postmask period as were covariables included in the adjusted model. When adjusting for traffic activity, total statewide caseload, public health complaints, and mean temperature, the daily caseload, hospital bed occupancy, ICU bed occupancy, ventilator occupancy, and daily mortality remained higher in the postmask period. CONCLUSIONS: There was no reduction in per-population daily mortality, hospital bed, ICU bed, or ventilator occupancy of COVID-19-positive patients attributable to the implementation of a mask-wearing mandate. [SEP]
            Title: Analysis of the Effects of COVID-19 Mask Mandates on Hospital Resource Consumption and Mortality at the County Level [SEP]
            Abstract: {example['input']} [SEP]
            Title:
            """
            example['answers'] = example['output']
            if self.tokenizer is not None:
                inputs = self.tokenizer(example['prompt'], padding=False, truncation=False)
                outputs = self.tokenizer(example['answers'], padding=False, truncation=False)
                example['input_ids'] = inputs['input_ids']
                example["attention_mask"] = inputs.attention_mask
                example["labels"] = outputs.input_ids.copy()
                example["labels"] = [-100 if _ == self.tokenizer.pad_token_id else _ for _ in example["labels"]]
            return example
        
        self.dataset = self.dataset.map(process_instance, load_from_cache_file=False)
        if self.tokenizer is not None:
            self.dataset.set_format(
                type="torch",
                columns=["input_ids", "attention_mask", "labels"],
                output_all_columns=True)
        
        return self.dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]
