<div align="center">
<h1>SVD-LLM: Singular Value Decomposition for Large Language Model Compression</h1>
</div>


## Introduction

### Key Designs
-  **Truncation-Aware Data Whitening**: Ensure truncating smaller singular values has lower compression loss. 
-  **Layer-Wise Closed-Form Update**: Compensate for accuracy degradation under high compression ratio.  

### Abstract
The advancements in Large Language Models (LLMs) have been hindered by their substantial sizes, which necessitate LLM compression methods for practical deployment. Singular Value Decomposition (SVD) offers a promis- ing solution for LLM compression. However, state-of-the-art SVD-based LLM compression methods have two key limitations: truncating smaller singular values may lead to higher compression loss, and the lack of update on the compressed weight after SVD truncation. In this work, we propose SVD-LLM, a new SVD-based LLM compression method that addresses the limitations of existing methods. SVD-LLM incorporates a truncation-aware data whitening strategy to ensure a direct mapping between singular values and compression loss. Moreover, SVD-LLM adopts a layer-wise closed-form model parameter update strategy to compensate for accuracy degrada- tion under high compression ratios. We evaluate SVD-LLM on a total of 10 datasets and eight models from three different LLM families at four differ- ent scales. Our results demonstrate the superiority of SVD-LLM over state- of-the-arts, especially at high model compression ratios. 

## Quick Start

### Installation
Please keep the version of the transformers package exactly equal to 4.35.2 since the svd-compressed version of LLM has a slight change of model structure (in the `component/.` folder).
```
pip install -r requirement.txt
```

### Quick Example
```
bash compress_llama.sh
```
This script would compress the LLaMA-7B model under 20\% and 50% compression ratio and automatically run the evaluation code, including both perplexity and efficiency of the compressed model.

    
## Step-by-Step Instructions  
    
We implement SVD-LLM with two different pipelines:
* Truncation-Aware Data Whitening + SVD Compression (used under **low** compression ratio)
* Truncation-Aware Data Whitening + SVD Compression + <u>Layer-Wise Closed-Form Update</u> (used under **high** compression ratio)
  
    
### 1. Truncation-Aware Data Whitening + SVD Compression
Under the low compression ratio (recommended ratio <= 0.3), we first run the data whitening of the LLM and saved the weight along with the whitening information.
```
python SVDLLM.py \
--step 1  \
--model HUGGINGFACE_MODEL_REPO \
--whitening_nsamples WHITENING_SAMPLE_NUMBER \
--dataset WHITENING_DATASET \
--seq_len MODEL_SEQ_LEN \
--save_path WHITENING_INFO_SAVING_PATH
```

We next load the whitening information and the weights to run SVD compression
```
python SVDLLM.py \
--step 2 \
--model_path WHITENING_INFO_SAVING_PATH \
--save_path COMPRESSD_MODEL_SAVING_PATH \
--ratio COMPRESSION_RATIO
```



### 2. Truncation-Aware Data Whitening + SVD Compression + Layer-Wise Closed-Form Update
Under the high compression ratio (recommended ratio > 0.3), we can further apply layer-wise closed-form update to update the weight matrix after the first pipeline to improve accuracy.

```
python SVDLLM.py \
--step 3 \
--model_path WHITENING_INFO_SAVING_PATH \
--save_path COMPRESSD_MODEL_SAVING_PATH \
--ratio COMPRESSION_RATIO
```