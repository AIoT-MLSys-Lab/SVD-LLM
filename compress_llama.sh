#!/bin/bash

# obtain the whitening matrix
python SVDLLM.py --model jeffwan/llama-7b-hf --step 1 --whitening_nsamples 256 --dataset wikitext2 --model_seq_len 2048 --save_path profile.pth

# SVDLLM under low compression ratio (ratio=0.8) with only data whitening
python SVDLLM.py --model  jeffwan/llama-7b-hf --model_path profile.pth --step 2 --save_path llama_7b_whitening_0.8.pth --ratio 0.8
# evaluate the perplexity of llama_7b_whitening_0.8.pth
python SVDLLM.py --model_path llama_7b_whitening_0.8.pth  --step 4
# evaluate the efficiency of llama_7b_whitening_0.8.pth
python SVDLLM.py --model_path llama_7b_whitening_0.8.pth  --step 5


# SVDLLM under high compression ratio (ratio=0.5) with both data whitening and closed-form update
python SVDLLM.py --model  jeffwan/llama-7b-hf --model_path profile.pth --step 3 --save_path llama_7b_whitening_0.5.pth --ratio 0.5
# evaluate the perplexity of llama_7b_whitening_0.3.pth
python SVDLLM.py --model_path llama_7b_whitening_0.5.pth  --step 4
# evaluate the efficiency of llama_7b_whitening_0.3.pth
python SVDLLM.py --model_path llama_7b_whitening_0.5.pth  --step 5