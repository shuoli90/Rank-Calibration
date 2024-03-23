# python compute_stats.py --model gpt-3.5-turbo --temperature 1.0 --dataset meadow --correctness rouge1
# python compute_stats.py --model gpt-3.5-turbo --temperature 1.0 --dataset meadow --correctness meteor
python compute_stats.py --model gpt-3.5-turbo --temperature 0.5 --dataset triviaqa --correctness rouge
python compute_stats.py --model gpt-3.5-turbo --temperature 1.5 --dataset triviaqa --correctness rouge
# python compute_stats.py --model 'meta-llama/Llama-2-7b-hf' --temperature 0.6 --dataset triviaqa --correctness meteor
# python compute_stats.py --model 'meta-llama/Llama-2-7b-hf' --temperature 0.6 --dataset squad --correctness meteor
# python compute_stats.py --model 'meta-llama/Llama-2-7b-hf' --temperature 0.6 --dataset nq-open --correctness meteor
# python compute_stats.py --model 'meta-llama/Llama-2-7b-hf' --temperature 0.6 --dataset triviaqa --correctness rouge1
# python compute_stats.py --model 'meta-llama/Llama-2-7b-hf' --temperature 0.6 --dataset squad --correctness rouge1
# python compute_stats.py --model 'meta-llama/Llama-2-7b-hf' --temperature 0.6 --dataset nq-open --correctness rouge1