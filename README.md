# **Uncertainty in Language Models: Assessment through Rank-Calibration**
<div align="center">

[![arXiv](https://img.shields.io/badge/cs.LG-arXiv%3A2310.03957-b31b1b)](https://arxiv.org/abs/2404.03163)

</div>

## Abstract
Language Models (LMs) have shown promising performance in natural language generation. However, as LMs often generate incorrect or hallucinated responses, it is crucial to correctly quantify their uncertainty in responding to given inputs. In addition to verbalized confidence elicited via prompting, many uncertainty measures (e.g., semantic entropy and affinity-graph-based measures) have been proposed. However, these measures can differ greatly, and it is unclear how to compare them, partly because they take values over different ranges (e.g., $[0,\infty)$ or $[0,1]$). In this work, we address this issue by developing a novel and practical framework, termed Rank-Calibration, to assess uncertainty and confidence measures for LMs. Our key tenet is that higher uncertainty (or lower confidence) should imply lower generation quality, on average. Rank-calibration quantifies deviations from this ideal relationship in a principled manner, without requiring ad hoc binary thresholding of the correctness score (e.g., ROUGE or METEOR). The broad applicability and the granular interpretability of our methods are demonstrated empirically.

<div align="center">

Indication Diagram via Rank-Calibration

</div>

<!-- ## Indication Diagram -->
![Indication](https://github.com/shuoli90/calibrate_framework/blob/main/figures/Indication.png)
*Indication diagrams* comparing two uncertainty measures, $U_{\rm NLL}$ (negative log-likelihood) and $U_{\rm Ecc}$ (eccentricity), for the GPT-3.5-turbo model on the TriviaQA benchmark. The ${\color{red} red}$ bars indicate the average correctness of different outputs, as a function of the corresponding relative uncertainty levels. The ${\color{blue} blue}$ and ${\color{#F67280} shallow \ red}$ areas
---deviating from the anti-diagonal line---indicate where the uncertainty measures are over-optimistic and pessimistic, respectively. Their sum is our *rank-miscalibration* metric:
![Indication](https://github.com/shuoli90/calibrate_framework/blob/main/figures/RCE.png)
<!-- $$\mathbf{E}_{U}\hspace{-2pt}\left[\left|\mathbf{P}_{U'}(\mathrm{reg}(U^\prime)\hspace{-1pt}\geq \hspace{-1pt}\mathrm{reg}(U))\right|
\right],$$ -->
where $U^\prime$ is an independent copy of $U$., which here is lower for $U_{\rm NLL}$ than $U_{\rm Ecc}$.

## Getting Started

- Create virtual environment using
```
python -m venv rce
pip install -r requirements.txt
```

- Before using OpenAI APIs, make sure you have the API key `OPENAI_API_KEY` updated in ./run/.env.