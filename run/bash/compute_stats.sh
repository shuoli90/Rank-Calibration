# python compute_stats.py --model 'meta-llama/Llama-2-7b-chat-hf' --temperature 0.6 --dataset triviaqa 
# python compute_stats.py --model 'meta-llama/Llama-2-7b-chat-hf' --temperature 0.6 --dataset squad 
# python compute_stats.py --model 'meta-llama/Llama-2-7b-chat-hf' --temperature 0.6 --dataset nq-open
# python compute_stats.py --model 'meta-llama/Llama-2-7b-chat-hf' --temperature 1.0 --dataset triviaqa
# python compute_stats.py --model 'meta-llama/Llama-2-7b-chat-hf' --temperature 1.0 --dataset squad
# python compute_stats.py --model 'meta-llama/Llama-2-7b-chat-hf' --temperature 1.0 --dataset nq-open
# python compute_stats.py --model 'meta-llama/Llama-2-7b-hf' --temperature 0.6 --dataset triviaqa
# python compute_stats.py --model 'meta-llama/Llama-2-7b-hf' --temperature 0.6 --dataset squad
# python compute_stats.py --model 'meta-llama/Llama-2-7b-hf' --temperature 0.6 --dataset nq-open
# python compute_stats.py --model mistralai/Mistral-7B-v0.1 --temperature 1.0 --dataset triviaqa
python compute_stats.py --model gpt-3.5-turbo --temperature 1.0 --dataset triviaqa
python compute_stats.py --model gpt-3.5-turbo --temperature 1.0 --dataset squad
python compute_stats.py --model gpt-3.5-turbo --temperature 1.0 --dataset nq-open
python compute_stats.py --model gpt-3.5-turbo --temperature 1.0 --dataset meadow

# python compute_stats.py --model 'meta-llama/Llama-2-7b-chat-hf' --temperature 0.6 --dataset triviaqa --correctness bert_similarity
# python compute_stats.py --model 'meta-llama/Llama-2-7b-chat-hf' --temperature 0.6 --dataset squad --correctness bert_similarity
# python compute_stats.py --model 'meta-llama/Llama-2-7b-chat-hf' --temperature 0.6 --dataset nq-open --correctness bert_similarity
# python compute_stats.py --model 'meta-llama/Llama-2-7b-chat-hf' --temperature 1.0 --dataset triviaqa --correctness bert_similarity
# python compute_stats.py --model 'meta-llama/Llama-2-7b-chat-hf' --temperature 1.0 --dataset squad --correctness bert_similarity
# python compute_stats.py --model 'meta-llama/Llama-2-7b-chat-hf' --temperature 1.0 --dataset nq-open --correctness bert_similarity
# python compute_stats.py --model 'meta-llama/Llama-2-7b-hf' --temperature 0.6 --dataset triviaqa --correctness bert_similarity
# python compute_stats.py --model 'meta-llama/Llama-2-7b-hf' --temperature 0.6 --dataset squad --correctness bert_similarity
# python compute_stats.py --model 'meta-llama/Llama-2-7b-hf' --temperature 0.6 --dataset nq-open --correctness bert_similarity
# python compute_stats.py --model mistralai/Mistral-7B-v0.1 --temperature 1.0 --dataset triviaqa --correctness bert_similarity
python compute_stats.py --model gpt-3.5-turbo --temperature 1.0 --dataset triviaqa --correctness bert_similarity
python compute_stats.py --model gpt-3.5-turbo --temperature 1.0 --dataset squad --correctness bert_similarity
python compute_stats.py --model gpt-3.5-turbo --temperature 1.0 --dataset nq-open --correctness bert_similarity
python compute_stats.py --model gpt-3.5-turbo --temperature 1.0 --dataset meadow --correctness bert_similarity