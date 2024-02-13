## Correctness measures:
- Rouge score
- Bleu score
- Meteor score
- Exact match
- Chatgpt-3.5/gpt-4 based

## Utility metrics:
- AUROC: Area Under the Receiver Operating Characteristic curve
- AUARC: Area under the Accuracy-Rejection Curve
- AUPRC-Positive: Area under the Precision-Recall Curve to identify correct samples
- AUPRC-Negative: Area under the Precision-Recall Curve to identify incorrect samples

## Calibration metrics:
- **Our new metric**
- ADPE: Adaptive debiased estimate for squared ECE
- ERCE: Expected rank-calibration error 
- **Baselines**
- PE: Plug-in estimate for ECE
- PE2: Plug-in estimate for squared ECE
- DPE: Debiased plugin estimate for squared ECE
