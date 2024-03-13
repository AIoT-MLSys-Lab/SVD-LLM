#coding:utf8
import os
import sys
import argparse
import torch.jit
from tqdm import tqdm
import torch
import torch.nn as nn

from utils.data_utils import *
from component.svd_llama import SVD_LlamaAttention, SVD_LlamaMLP
from component.svd_mistral import SVD_MistralAttention, SVD_MistralMLP
from utils.model_utils import *
from evaluater import * 

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)

@torch.no_grad()
def profle_aat(name, model, calib_loader, dev):
    model.eval()
    if "llama" in name or "mistral" in name or "vicuna" in name:
        layers = model.model.layers
    elif "opt" in name:
        layers = model.model.decoder.layers
    # print("Obtaining Whitening Matrix...")
    def hook(module, input, output):
        inp = input[0].detach().float()
        # if "opt" in name:
        if inp.dim() == 2:
            inp = inp.unsqueeze(0)
        adds = torch.matmul(inp.transpose(1,2), inp)
        adds_sum = torch.sum(adds, dim=0)
        module.scaling_diag_matrix += adds_sum
        del inp, adds, adds_sum
        torch.cuda.empty_cache()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.scaling_diag_matrix = 0
            module.register_forward_hook(hook)
    for batch in tqdm(calib_loader):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        model(**batch)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module._forward_hooks.clear()
    torch.cuda.empty_cache()
    model = model.cpu()
    print("Start Converting...")
    for i in tqdm(range(len(layers))):
        layer = layers[i].to(dev)
        subset = find_layers(layer)
        for name in subset:
            W = subset[name].weight.data.float().cuda()
            raw_scaling_diag_matrix = subset[name].scaling_diag_matrix.float().cuda()
            try:
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
            except Exception as e:
                print("Warning: eigen scaling_diag_matrix is not positive!")
                if torch.isnan(raw_scaling_diag_matrix).any():
                    print("Warning: scaling_diag_matrix contains NaN!")
                elif torch.isinf(raw_scaling_diag_matrix).any():
                    print("Warning: scaling_diag_matrix contains Inf!")
                if not torch.equal(raw_scaling_diag_matrix, raw_scaling_diag_matrix.T):
                    print("Warning: scaling_diag_matrix is not a symmetric matrix!")
                eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
                raw_scaling_diag_matrix += (- eigenvalues[0] + 1e-6) * torch.eye(raw_scaling_diag_matrix.shape[0]).cuda()
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
            try:
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            except Exception as e:
                print("Warning: scaling_diag_matrix is not full rank!")
                scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0]).cuda() 
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            W_scale = torch.matmul(W, scaling_diag_matrix)
            subset[name].u, subset[name].s, subset[name].vt = torch.linalg.svd(W_scale, full_matrices=False)
            subset[name].vt = torch.matmul(subset[name].vt, scaling_matrix_inv)
            W_scale = scaling_matrix_inv = scaling_diag_matrix = raw_scaling_diag_matrix = None
            del W_scale, scaling_matrix_inv, scaling_diag_matrix, raw_scaling_diag_matrix
            torch.cuda.empty_cache()
        layers[i] = layer.cpu()
        

@torch.no_grad()
def profle_aat_large_scale(name, model, calib_loader, dev):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    if "llama" in name or "mistral" in name or "vicuna" in name:
        layers = model.model.layers
    elif "opt" in name:
        layers = model.model.decoder.layers
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (len(calib_loader), model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            if cache['attention_mask'] is None:
                cache['attention_mask'] = kwargs['attention_mask']
                cache['position_ids'] = kwargs['position_ids']
            else:
                cache['attention_mask'] = torch.cat((cache['attention_mask'], kwargs['attention_mask']), dim=0)
                cache['position_ids'] = torch.cat((cache['position_ids'], kwargs['position_ids']), dim=0)
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in calib_loader:
        try:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            model(**batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_masks = cache['attention_mask']
    position_ids = cache['position_ids']
    for i in tqdm(range(len(layers))):
        layer = layers[i].to(dev)
        subset = find_layers(layer)        
        def hook(module, input, output):
            inp = input[0].detach().float()
            # if "opt" in name:
            if inp.dim() == 2:
                inp = inp.unsqueeze(0)
            adds = torch.matmul(inp.transpose(1,2), inp)
            adds_sum = torch.sum(adds, dim=0)
            module.scaling_diag_matrix += adds_sum
            del inp, adds, adds_sum, output
            torch.cuda.empty_cache()
        handles = []
        for name in subset:
            subset[name].scaling_diag_matrix = 0
            handles.append(subset[name].register_forward_hook(hook))
        for j in range(inps.shape[0]):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_masks[j].unsqueeze(0), position_ids=position_ids[j].unsqueeze(0))[0]
        for h in handles:
            h.remove()
        layer = layer.cpu()
        for name in subset:
            subset[name].scaling_diag_matrix = subset[name].scaling_diag_matrix.cpu()
        torch.cuda.empty_cache()
        for name in subset:
            W = subset[name].weight.data.float().cuda()
            raw_scaling_diag_matrix = subset[name].scaling_diag_matrix.double().cuda()
            try:
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix).float()
            except Exception as e:
                print("Warning: eigen scaling_diag_matrix is not positive!")
                if torch.isnan(raw_scaling_diag_matrix).any():
                    print("Warning: raw scaling_diag_matrix contains NaN!")
                elif torch.isinf(raw_scaling_diag_matrix).any():
                    print("Warning: raw scaling_diag_matrix contains Inf!")
                if not torch.equal(raw_scaling_diag_matrix, raw_scaling_diag_matrix.T):
                    print("Warning: raw scaling_diag_matrix is not a symmetric matrix!")
                eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
                raw_scaling_diag_matrix += (- eigenvalues[0] + 1e-3) * torch.eye(raw_scaling_diag_matrix.shape[0]).cuda()
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix).float()
                if torch.isnan(scaling_diag_matrix).any():
                    print("Warning: scaling_diag_matrix contains NaN!")
                elif torch.isinf(scaling_diag_matrix).any():
                    print("Warning: scaling_diag_matrix contains Inf!")
                del eigenvalues
            try:
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            except Exception as e:
                print("Warning: scaling_diag_matrix is not full rank!")
                reg_inv =  1e-3 * torch.eye(scaling_diag_matrix.shape[0]).cuda() 
                scaling_diag_matrix += reg_inv
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
                del reg_inv
            W_scale = torch.matmul(W, scaling_diag_matrix)
            u, s, vt = torch.linalg.svd(W_scale, full_matrices=False)
            subset[name].u, subset[name].s = u.cpu(), s.cpu()
            subset[name].vt = torch.matmul(vt, scaling_matrix_inv).cpu()
            W_scale = scaling_matrix_inv = scaling_diag_matrix = raw_scaling_diag_matrix = u = s = vt = None
            del W_scale, scaling_matrix_inv, scaling_diag_matrix, raw_scaling_diag_matrix, u, s, vt
            torch.cuda.empty_cache()
        # layer = layer.to(dev)
        # for id in range(inps.shape[0]):
        #     outs[id] = layer(inps[id], attention_mask=attention_masks, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        inps = outs
        torch.cuda.empty_cache()
    model.config.use_cache = use_cache
     
 
@torch.no_grad()
def whitening(model_name, model, ratio):
    model.eval()
    layers = model.model.layers
    print("Start AAT decomposition...")
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        #### Replace Attn, MLP ####
        if "llama" in model_name or "vicuna" in model_name:
            svd_attn = SVD_LlamaAttention(config=model.config, ratio=[ratio, ratio, ratio, ratio])
            svd_mlp = SVD_LlamaMLP(hidden_size=layer.hidden_size, intermediate_size=model.config.intermediate_size, hidden_act=model.config.hidden_act, ratio=[ratio,ratio,ratio])
        elif "mistral" in model_name:
            svd_attn = SVD_MistralAttention(config=model.config, ratio=[ratio, ratio, ratio, ratio])
            svd_mlp = SVD_MistralMLP(config=model.config, ratio=[ratio,ratio,ratio])
        #### Replace Attn, MLP ####
        for name in subset:
            W = subset[name].weight.data.float()
            U, S, VT = subset[name].u, subset[name].s, subset[name].vt
            num_s_after_trunc = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))
            truc_s = S[:num_s_after_trunc]
            truc_u = U[:, :num_s_after_trunc]
            truc_v = VT[:num_s_after_trunc, :]
            truc_sigma = torch.diag(truc_s)
            #### Replace Attn, MLP ####
            sqrtSigma = torch.sqrt(truc_sigma)
            svd_u = torch.matmul(truc_u, sqrtSigma)
            svd_v = torch.matmul(sqrtSigma, truc_v)
            if "q_proj" in name:
                svd_attn.q_u_proj.weight.data = svd_u
                svd_attn.q_v_proj.weight.data = svd_v
            elif "k_proj" in name:
                svd_attn.k_u_proj.weight.data = svd_u
                svd_attn.k_v_proj.weight.data = svd_v
            elif "v_proj" in name:
                svd_attn.v_u_proj.weight.data = svd_u
                svd_attn.v_v_proj.weight.data = svd_v
            elif "o_proj" in name:
                svd_attn.o_u_proj.weight.data = svd_u
                svd_attn.o_v_proj.weight.data = svd_v
                layer.self_attn =  svd_attn
            elif "gate_proj" in name:
                svd_mlp.gate_u_proj.weight.data = svd_u
                svd_mlp.gate_v_proj.weight.data = svd_v
            elif "down_proj" in name:
                svd_mlp.down_u_proj.weight.data = svd_u
                svd_mlp.down_v_proj.weight.data = svd_v
            elif "up_proj" in name:
                svd_mlp.up_u_proj.weight.data = svd_u
                svd_mlp.up_v_proj.weight.data = svd_v
                layer.mlp = svd_mlp
        print("Layer ", i, " done")
        del layer
        torch.cuda.empty_cache()
        

@torch.no_grad()
def whitening_local_update(model_name, model, dataloader, ratio, dev, direct_update=False):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (len(dataloader), model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            if cache['attention_mask'] is None:
                cache['attention_mask'] = kwargs['attention_mask']
                cache['position_ids'] = kwargs['position_ids']
            else:
                cache['attention_mask'] = torch.cat((cache['attention_mask'], kwargs['attention_mask']), dim=0)
                cache['position_ids'] = torch.cat((cache['position_ids'], kwargs['position_ids']), dim=0)
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_masks = cache['attention_mask']
    position_ids = cache['position_ids']
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        subset = find_layers(layer)
        gpts = {}
        if "llama" in model_name or "vicuna" in model_name:
            svd_attn = SVD_LlamaAttention(config=model.config, ratio=[ratio, ratio, ratio, ratio])
            svd_mlp = SVD_LlamaMLP(hidden_size=layer.hidden_size, intermediate_size=model.config.intermediate_size, hidden_act=model.config.hidden_act, ratio=[ratio,ratio,ratio])
        elif "mistral" in model_name:
            svd_attn = SVD_MistralAttention(config=model.config, ratio=[ratio, ratio, ratio, ratio])
            svd_mlp = SVD_MistralMLP(config=model.config, ratio=[ratio,ratio,ratio])
        for name in subset:
            gpts[name] = local_update(subset[name], ratio=ratio, name=name, direct_update=direct_update)
        
        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch_update_u(inp[0].data, out.data)
            return tmp
        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        outs = layer(inps, attention_mask=attention_masks, position_ids=position_ids)[0]
        for h in handles:
            h.remove()
        for name in gpts:
            svd_u, svd_v = gpts[name].fasterprune()
            if "q_proj" in name:
                svd_attn.q_u_proj.weight.data = svd_u
                svd_attn.q_v_proj.weight.data = svd_v
            elif "k_proj" in name:
                svd_attn.k_u_proj.weight.data = svd_u
                svd_attn.k_v_proj.weight.data = svd_v
            elif "v_proj" in name:
                svd_attn.v_u_proj.weight.data = svd_u
                svd_attn.v_v_proj.weight.data = svd_v
            elif "o_proj" in name:
                svd_attn.o_u_proj.weight.data = svd_u
                svd_attn.o_v_proj.weight.data = svd_v
                layer.self_attn =  svd_attn
            elif "gate_proj" in name:
                svd_mlp.gate_u_proj.weight.data = svd_u
                svd_mlp.gate_v_proj.weight.data = svd_v
            elif "down_proj" in name:
                svd_mlp.down_u_proj.weight.data = svd_u
                svd_mlp.down_v_proj.weight.data = svd_v
            elif "up_proj" in name:
                svd_mlp.up_u_proj.weight.data = svd_u
                svd_mlp.up_v_proj.weight.data = svd_v
                layer.mlp = svd_mlp
        layer = layer.to(dev)
        outs = layer(inps, attention_mask=attention_masks, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del gpts
        torch.cuda.empty_cache()
        inps = outs
        outs = None
        del outs
    model.config.use_cache = use_cache


class local_update:
    def __init__(self, layer, ratio, name, direct_update=False):
        self.layer = layer
        self.name = name
        self.dev = self.layer.weight.device
        # W = layer.weight.data.clone()
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        if direct_update:
            self.U, self.S, self.VT = torch.linalg.svd(W.data, full_matrices=False)
        else:   
            self.U, self.S, self.VT = layer.u, layer.s, layer.vt
        # trucation SVD
        num_s_after_trunc = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))
        self.truc_s = self.S[:num_s_after_trunc].cuda()
        self.truc_u = self.U[:, :num_s_after_trunc].cuda()
        self.truc_v = self.VT[:num_s_after_trunc, :].cuda()
        self.truc_sigma = torch.diag(self.truc_s)
        self.new_w = torch.matmul(self.truc_u, torch.matmul(self.truc_sigma, self.truc_v[:num_s_after_trunc, :]))
        # intialize H for close form solution
        self.updated_err = self.error = 0

    def add_batch_update_u(self, inp, out):
        inps = inp.view(inp.shape[0] * inp.shape[1], inp.shape[2]).float()
        outs = out.view(out.shape[0] * out.shape[1], out.shape[2]).float()
        new_w = torch.matmul(self.truc_u, torch.matmul(self.truc_sigma, self.truc_v))
        new_output = inps.matmul(new_w.t())
        self.error = torch.sqrt(torch.sum((outs - new_output)**2)).item() / torch.norm(outs, p='fro').item()
        # print(f"truncted error: {self.error}")
        x =  torch.matmul(torch.matmul(inps, self.truc_v.T), self.truc_sigma)
        self.updated_uT = torch.linalg.lstsq(x,outs).solution
        updated_output = torch.matmul(torch.matmul(torch.matmul(inps, self.truc_v.T), self.truc_sigma), self.updated_uT)
        self.updated_error = torch.sqrt(torch.sum((outs - updated_output)**2)).item() / torch.norm(outs, p='fro').item()
        # print(f"updated error: {self.updated_error}")
        inps = outs = new_output = updated_output = x = new_w = None
        del inps, outs, new_output, updated_output, x, new_w
        torch.cuda.empty_cache()
        # print(f"Finish {self.name}"
    
    def fasterprune(self):
        sqrtSigma = torch.sqrt(self.truc_sigma)
        self.appendU = self.updated_uT.t().matmul(sqrtSigma)
        self.appendV = sqrtSigma.matmul(self.truc_v)
        return self.appendU, self.appendV


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model', type=str, default='jeffwan/llama-7b-hf',
        help='LLaMA model to load, pass `jeffwan/llama-7b-hf`'
    )
    parser.add_argument(
        '--model_path', type=str, default=None,
        help='local compressed model path or whitening information path'
    )
    parser.add_argument(
        '--ratio', type=float, default=0.8,
        help='Target compression ratio,(0,1), default=0.8, means only keeping about 80% of the params.'
    )
    parser.add_argument(
        '--dataset', type=str, default='wikitext2',
        help='Where to extract calibration data from [wikitext2, ptb, c4]'
    )
    parser.add_argument(
        '--whitening_nsamples', type=int, default=256,
        help='Number of calibration data samples for whitening.'
    )
    parser.add_argument(
        '--updating_nsamples', type=int, default=16,
        help='Number of calibration data samples for udpating.'
    )
    parser.add_argument(
        '--save_path', type=str, default=None,
        help='the path to save the compressed model checkpoints.`'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data'
    )
    parser.add_argument(
        '--DEV', type=str, default="cuda", 
        help='device'
    )
    parser.add_argument(
        '--model_seq_len', type=int, default=2048, 
        help='the default sequence length of the LLM'
    )
    parser.add_argument(
        '--eval_batch_size', type=int, default=32,
        help='inference bactch size'
    )
    parser.add_argument(
        '--gen_seq_len', type=int, default=1024, 
        help='generated sequence len for efficiency evaluation'
    )
    parser.add_argument(
        '--step', type=int, default=4, 
        help='the step to run the compression'
    )
    parser.add_argument(
        '--lora', type=str, default=None,
        help='the lora updated weight path to run the accuracy evaluation`'
    )
    
    args = parser.parse_args()
    if args.step == 1:
        model, tokenizer = get_model_from_huggingface(args.model)
        model = model.to(args.DEV)
        model = model.eval()
        cali_white_data = get_calib_train_data(args.dataset, tokenizer, args.whitening_nsamples, seqlen=args.model_seq_len)
        profle_aat(args.model, model, cali_white_data, args.DEV)
    elif args.step == 2:
        model, tokenizer = get_model_from_local(args.model_path)
        model = model.to(args.DEV)
        model.eval()
        whitening(args.model, model, args.ratio)
    elif args.step == 3:
        model, tokenizer = get_model_from_local(args.model_path)
        model = model.to(args.DEV)
        model.eval()
        model = model.float()
        dataloader, _ = get_loaders(args.dataset, nsamples=args.updating_nsamples, seed=args.seed, model=args.model, seqlen=args.model_seq_len)
        whitening_local_update(args.model_path, model, dataloader, args.ratio, args.DEV)
    elif args.step >= 4:
        print(f"evaluating {args.model_path}...")
        if args.model_path == "original":
            model, tokenizer = get_model_from_huggingface(args.model)
        else:
            model, tokenizer = get_model_from_local(args.model_path)
            if args.lora is not None:
                from utils.peft import PeftModel
                model = PeftModel.from_pretrained(
                    model,
                    args.lora,
                    torch_dtype=torch.float16,
                )
                model = model.merge_and_unload()
        model.eval()
        is_fp16 = all(param.dtype == torch.float16 for param in model.parameters())
        print(f"is_fp16:{is_fp16}")
        if not is_fp16 and args.DEV != "cpu":
            model.half()
        elif args.DEV == "cpu":
            model.float()
        model = model.to(args.DEV)
        if args.step == 4:
            ppl_eval(model, tokenizer, datasets=['wikitext2'], model_seq_len=args.model_seq_len, batch_size=args.eval_batch_size, device=args.DEV)
        elif args.step == 5:
            eff_eval(model, tokenizer, generated_len=args.gen_seq_len, batch_size=args.eval_batch_size, device=args.DEV)
    if args.save_path is not None:
        is_fp16 = all(param.dtype == torch.float16 for param in model.parameters())
        if not is_fp16 and args.DEV != "cpu":
            model.half()    
        torch.save({'model': model, 'tokenizer': tokenizer}, args.save_path)