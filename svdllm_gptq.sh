#!/bin/bash

# run data whitening with 20% compression ratio
python SVDLLM.py --model jeffwan/llama-7b-hf --step 1 --ratio 0.2 --whitening_nsamples 256 --dataset wikitext2 --seed 3 --model_seq_len 2048 --save_path .

# further compress the model with GPTQ-4bit
python quant_llama.py --model_path whitening/jeffwan_llama_7b_hf_whitening_0.2.pt --dataset c4 --wbits 4 --true-sequential --act-order --new-eval  --save svdllm_gptq_4.pt