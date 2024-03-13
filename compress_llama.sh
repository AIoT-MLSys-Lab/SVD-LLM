#!/bin/bash

# obtain the whitening matrix
python SVDLLM.py --model jeffwan/llama-7b-hf --step 1 --whitening_nsamples 256 --dataset wikitext2 --model_seq_len 2048 --save_path profile.pt

# SVDLLM under low compression ratio (ratio=0.8) with only data whitening
python SVDLLM.py --step 2 --ratio 0.8 --model jeffwan/llama-7b-hf --model_path profile.pt  --save_path llama_7b_whitening_0.8.pt 
# evaluate the perplexity of llama_7b_whitening_0.8.pt
python SVDLLM.py --step 4 --model_path llama_7b_whitening_0.8.pt
# evaluate the efficiency of llama_7b_whitening_0.8.pt
python SVDLLM.py --step 5 --model_path llama_7b_whitening_0.8.pt


# SVDLLM under high compression ratio (ratio=0.5) with both data whitening and closed-form update
python SVDLLM.py --step 3 --ratio 0.5 --model jeffwan/llama-7b-hf --model_path profile.pt --save_path llama_7b_whitening_0.5.pt 
# evaluate the perplexity of llama_7b_whitening_0.3.pt
python SVDLLM.py --step 4 --model_path llama_7b_whitening_0.5.pt
# evaluate the efficiency of llama_7b_whitening_0.3.pt
python SVDLLM.py --step 5 --model_path llama_7b_whitening_0.5.pt