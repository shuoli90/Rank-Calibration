# python plot_graphs.py --model 'meta-llama/Llama-2-7b-chat-hf' --temperature 0.6 --dataset triviaqa --correctness rouge1
# python plot_graphs.py --model 'meta-llama/Llama-2-7b-chat-hf' --temperature 0.6 --dataset squad --correctness rouge1
# python plot_graphs.py --model 'meta-llama/Llama-2-7b-chat-hf' --temperature 0.6 --dataset nq-open --correctness rouge1
# python plot_graphs.py --model 'meta-llama/Llama-2-7b-chat-hf' --temperature 1.0 --dataset triviaqa --correctness rouge1
# python plot_graphs.py --model 'meta-llama/Llama-2-7b-chat-hf' --temperature 1.0 --dataset squad --correctness rouge1
# python plot_graphs.py --model 'meta-llama/Llama-2-7b-chat-hf' --temperature 1.0 --dataset nq-open --correctness rouge1
# python plot_graphs.py --model 'meta-llama/Llama-2-7b-hf' --temperature 0.6 --dataset triviaqa --correctness rouge1
# python plot_graphs.py --model 'meta-llama/Llama-2-7b-hf' --temperature 0.6 --dataset squad --correctness rouge1
# python plot_graphs.py --model 'meta-llama/Llama-2-7b-hf' --temperature 0.6 --dataset nq-open --correctness rouge1
# python plot_graphs.py --model mistralai/Mistral-7B-v0.1 --temperature 1.0 --dataset triviaqa --correctness rouge1
python plot_graphs.py --model gpt-3.5-turbo --temperature 1.5 --dataset triviaqa --correctness rouge1
python plot_graphs.py --model gpt-3.5-turbo --temperature 0.5 --dataset triviaqa --correctness rouge1
python plot_graphs.py --model gpt-3.5-turbo --temperature 1.5 --dataset triviaqa --correctness meteor
python plot_graphs.py --model gpt-3.5-turbo --temperature 0.5 --dataset triviaqa --correctness meteor
# python plot_graphs.py --model gpt-3.5-turbo --temperature 1.0 --dataset nq-open --correctness rouge1
# python plot_graphs.py --model gpt-3.5-turbo --temperature 1.0 --dataset meadow --correctness rouge1