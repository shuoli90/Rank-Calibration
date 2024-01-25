# calibrate_framework
This is the repo for constructing a comprehensive and rigorous evaluation framework for LLM calibration.

## Structure
We structure the implementation by functionality. Here is a brief description of folders:
- datasets: this folder contains scripts for pre-processing nlp datasets. Each script contains preprocessing steps for datasets under the same nlp task. The list of nlp tasks and corresponding datasets are listed in README file.
- models: this folder prepare opensource models and OpenAI APIs. The preparation operations include: 1, load in models or setup API; 2, setup generation pipeline. Users are supposed to bring their own OpenAI account.
- metrics: this folder contains implementations for generation correctness, calibration metrics, and statistical test implementations. We should include a table listing all available metrics.
- indicators: this folder contains implementations for white-box and black-box confidence indicators. We should include a table listing all available indicators.
- utils: this folder contains miscellaneous functions.
- run: this folder contains scripts that are supposed to be run by users. Functions include generating responses, calibrating LLM using a specific indicator, comparing multiple indicators.