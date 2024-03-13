#coding:utf8
import os
import sys
import torch
import torch.nn as nn

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)

# bandaid fix
dev = torch.device("cuda")

def get_model_from_huggingface(model_id):
    from transformers import AutoModelForCausalLM, LlamaTokenizer, AutoTokenizer, LlamaForCausalLM
    if "opt" in model_id or "mistral" in model_id:
        tokenizer = AutoTokenizer.from_pretrained(model_id, device_map="cpu", trust_remote_code=True)
    else:
        tokenizer = LlamaTokenizer.from_pretrained(model_id, device_map="cpu", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu", torch_dtype=torch.float16, trust_remote_code=True, cache_dir='/fs/scratch/PAS2473/huggingface')
    model.seqlen = 2048
    return model, tokenizer

def get_model_from_local(model_id):
    pruned_dict = torch.load(model_id, map_location='cpu')
    tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model']
    return model, tokenizer

def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


@torch.no_grad()
def load_sensitivity(model_id, scaling_method="abs_mean", alpha=0.5, n_calib_samples=32, calib_dataset="wikitext2"):
    cache_file = f"cache/{model_id.replace('/','_')}_sensitivity_{scaling_method}_{alpha}_{n_calib_samples}_{calib_dataset}.pt"
    sensitivity_dict = torch.load(cache_file, map_location="cpu")
    return sensitivity_dict

def calculate_compression_ratio(sensitivity_dict, module_dict, ratio_target):
    sensitivity_list = []
    for layername, v in sensitivity_dict.items():
        for ratio, ppl in v.items():
            sensitivity_list.append((layername, ratio, ppl))
    sorted_sensitive_list = sorted(sensitivity_list, key=lambda x: -x[2])

    # binary search
    high = len(sorted_sensitive_list) - 1
    low = 0
    while low < high:
        mid = (low + high) // 2
        layers_min_ratio = {layername: 1 for layername in sensitivity_dict.keys()}
        for layername, ratio, ppl in sorted_sensitive_list[mid:]:
            layers_min_ratio[layername] = min(layers_min_ratio[layername], ratio)
        tot_params = 0
        compress_params = 0
        for layername, ratio in layers_min_ratio.items():
            raw_linear = module_dict[layername]
            tot_params += raw_linear.weight.numel()
            compress_params += raw_linear.weight.numel() * ratio
        param_ratio = compress_params / tot_params
        msg = f"low={low} mid={mid}, high={high}, param_ratio={param_ratio}({compress_params}/{tot_params})"
        print(msg)
        if param_ratio > ratio_target:
            high = mid
        else:
            low = mid + 1
    layers_min_ratio = {layername: 1 for layername in sensitivity_dict.keys()}
    for layername, ratio, ppl in sorted_sensitive_list[mid:]:
        layers_min_ratio[layername] = min(layers_min_ratio[layername], ratio)
    return layers_min_ratio