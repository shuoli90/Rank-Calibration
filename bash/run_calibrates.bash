# CUDA_VISIBLE_DEVICES=5 python ../run/calibrate.py --model meta-llama/Llama-2-7b-hf
CUDA_VISIBLE_DEVICES=2 python run/calibrate.py --model facebook/opt-350m
# CUDA_VISIBLE_DEVICES=5 python ../run/calibrate.py --model meta-llama/Llama-2-7b-hf --indicator "generation_probability"