for CORRECT in rouge rouge1 meteor bert_similarity
do
    python compute_stats_tmp.py --model 'meta-llama/Llama-2-7b-chat-hf' --temperature 0.6 --dataset triviaqa --correctness $CORRECT
    python compute_stats_tmp.py --model 'meta-llama/Llama-2-7b-chat-hf' --temperature 0.6 --dataset squad  --correctness $CORRECT
    python compute_stats_tmp.py --model 'meta-llama/Llama-2-7b-chat-hf' --temperature 0.6 --dataset nq-open --correctness $CORRECT
    python compute_stats_tmp.py --model 'meta-llama/Llama-2-7b-chat-hf' --temperature 1.0 --dataset triviaqa --correctness $CORRECT
    python compute_stats_tmp.py --model 'meta-llama/Llama-2-7b-chat-hf' --temperature 1.0 --dataset squad --correctness $CORRECT
    python compute_stats_tmp.py --model 'meta-llama/Llama-2-7b-chat-hf' --temperature 1.0 --dataset nq-open --correctness $CORRECT
    python compute_stats_tmp.py --model 'meta-llama/Llama-2-7b-hf' --temperature 0.6 --dataset triviaqa --correctness $CORRECT
    python compute_stats_tmp.py --model 'meta-llama/Llama-2-7b-hf' --temperature 0.6 --dataset squad --correctness $CORRECT   
    python compute_stats_tmp.py --model 'meta-llama/Llama-2-7b-hf' --temperature 0.6 --dataset nq-open --correctness $CORRECT
    python compute_stats_tmp.py --model gpt-3.5-turbo --temperature 1.0 --dataset triviaqa --correctness $CORRECT
    python compute_stats_tmp.py --model gpt-3.5-turbo --temperature 1.0 --dataset squad --correctness $CORRECT
    python compute_stats_tmp.py --model gpt-3.5-turbo --temperature 1.0 --dataset nq-open --correctness $CORRECT
    python compute_stats_tmp.py --model gpt-3.5-turbo --temperature 1.0 --dataset meadow --correctness $CORRECT
    python compute_stats_tmp.py --model gpt-3.5-turbo --temperature 1.5 --dataset triviaqa --correctness $CORRECT
    python compute_stats_tmp.py --model gpt-3.5-turbo --temperature 0.5 --dataset triviaqa --correctness $CORRECT
done