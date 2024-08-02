import torch
import numpy as np
import einops
import gc
from functools import partial
from tqdm import tqdm

import plotly.express as px

from transformer_lens import HookedTransformer, ActivationCache
from transformer_lens import utils

class AtPTransformer(HookedTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def attribution_patching(self, x_clean, x_corr, a_clean, a_corr, component, prepend_bos=True, method='standard', num_alphas=5, n_last_tokens=128):
        if isinstance(x_clean, str):
            clean_tokens = self.to_tokens(x_clean, prepend_bos=prepend_bos)
        else:
            clean_tokens = x_clean

        if isinstance(x_corr, str):
            corr_tokens = self.to_tokens(x_corr, prepend_bos=prepend_bos)
        else:
            corr_tokens = x_corr

        if isinstance(a_clean, str):
            a_clean = self.to_single_token(a_clean)

        if isinstance(a_corr, str):
            a_corr = self.to_single_token(a_corr)

        with torch.no_grad():
            corr_logits, corr_cache = self.get_cache_fw(corr_tokens, component)

        if method == 'standard':
            logits_diff_, clean_cache, clean_grad_cache = self.get_cache_fw_and_bw(clean_tokens, a_clean, a_corr, corr_logits, component=component)
        elif method == 'ig':
            with torch.no_grad():
                clean_logits, clean_cache = self.get_cache_fw(clean_tokens, component)

        clean_cache = ActivationCache(clean_cache, self).to('cpu')
        corr_cache = ActivationCache(corr_cache, self).to('cpu')

        corr_act = clean_cache.stack_activation(component)[:, 0, -n_last_tokens:]
        clean_act = corr_cache.stack_activation(component)[:, 0, -n_last_tokens:] # comp, pos dm
        del clean_cache, corr_cache
        
        if clean_act.ndim > 3:
            clean_act = clean_act.transpose(1, 2)
            corr_act = corr_act.transpose(1, 2)
            clean_act = clean_act.reshape(-1, clean_act.size(2), clean_act.size(3))
            corr_act = corr_act.reshape(-1, corr_act.size(2), corr_act.size(3))
            
        if method == 'standard':
            clean_grad_cache = ActivationCache(clean_grad_cache, self).to('cpu')
            clean_grad_act = clean_grad_cache.stack_activation(component).squeeze()
            if clean_grad_act.ndim > 3:
                clean_grad_act = clean_grad_act.transpose(1, 2)
                clean_grad_act = clean_grad_act.reshape(-1, clean_grad_act.size(2), clean_grad_act.size(3))
            clean_grad_act = clean_grad_act[:, -n_last_tokens:].cpu()
        elif method == 'ig':
            clean_grad_act = []
            alphas = torch.linspace(0, 1, num_alphas)
            k = clean_act.shape[0] // self.cfg.n_layers
            for l in tqdm(range(self.model.cfg.n_layers)):
                ig_patch = torch.zeros_like(clean_act[k*l:k*(l+1)], device=clean_act.device)
                for alpha in alphas:
                    a_alpha = alpha * clean_act[k*l:k*(l+1)] + (1 - alpha) * corr_act[k*l:k*(l+1)]
                    logits_alpha, grad_alpha = self.get_cache_fw_with_modified_activations(clean_tokens, a_alpha, a_clean, a_corr, l, component)
                    if grad_alpha.ndim > 3:
                        grad_alpha = grad_alpha.reshape(-1, grad_alpha.size(1), grad_alpha.size(3))
                    grad_alpha = grad_alpha[:, -n_last_tokens:].cpu()
                    ig_patch += grad_alpha * (clean_act[k*l:k*(l+1)] - corr_act[k*l:k*(l+1)])
                    del a_alpha, logits_alpha, grad_alpha
                clean_grad_act.append(ig_patch / num_alphas)
                torch.cuda.empty_cache()
                gc.collect()
            clean_grad_act = torch.cat(clean_grad_act, dim=0)

        print("Gradients collected! Computing the patch...")
        patch = einops.reduce(
            clean_grad_act * (corr_act - clean_act),
            "component pos d_model -> component pos",
            "sum",
        )
        del clean_act, corr_act, clean_grad_act
        torch.cuda.empty_cache()

        return patch

    def get_cache_fw(self, tokens, components):
        filter = lambda name: any([utils.get_act_name(c) in name for c in components])

        self.reset_hooks()

        cache = {}
        def fw_cache_hook(act, hook):
            cache[hook.name] = act.detach()

        self.add_hook(filter, fw_cache_hook, "fwd")
        logits = self.forward(tokens)
        self.reset_hooks()
        return logits, ActivationCache(cache, self)

    def get_cache_fw_and_bw(self, tokens, a_clean, a_corr, corr_logits, component='all'):
        if component == 'all':
            filter = lambda name: "_input" not in name
        elif component == 'qkv':
            filter = lambda name: name.split('.')[-1].strip() in ['hook_q', 'hook_k', 'hook_v'] and "_input" not in name
        else:
            filter = lambda name: component in name
            
        self.reset_hooks()
        
        cache = {}
        def fw_cache_hook(act, hook):
            cache[hook.name] = act.detach()

        self.add_hook(filter, fw_cache_hook, "fwd")
        
        grad_cache = {}
        def bw_cache_hook(act, hook):
            grad_cache[hook.name] = act.detach()
        
        self.add_hook(filter, bw_cache_hook, "bwd")

        clean_logits = self.forward(tokens).cpu()
        value = self.logits_diff(clean_logits, a_clean, a_corr) #- logits_diff(corr_logits.cpu(), a_clean, a_corr)
        value.backward()
        
        self.reset_hooks()
        return (
            value.item(),
            ActivationCache(cache, self),
            ActivationCache(grad_cache, self),
        )

    def get_cache_fw_with_modified_activations(self, tokens, x_int, a_clean, a_corr, layer, component):
        hook_point = utils.get_act_name(component, layer)
        self.reset_hooks()
        
        def fw_hook(act, mod_act, hook):
            act = mod_act

        fw_hook_fn = partial(fw_hook, mod_act=x_int.squeeze())
        self.add_hook(hook_point, fw_hook_fn, "fwd")
        
        grad_cache = {}
        def bw_cache_hook(act, hook):
            grad_cache[hook.name] = act.detach()
        
        self.add_hook(hook_point, bw_cache_hook, "bwd")
        logits = self.forward(tokens)
        value = self.logits_diff(logits, a_corr, a_clean)
        value.backward()
        
        self.reset_hooks()
        return value.item(), grad_cache[hook_point]

    def logits_diff(self, logits, a_clean, a_corr=None):
        if isinstance(a_clean, str):
            a_clean = self.to_single_token(a_clean)
        if a_corr:
            if isinstance(a_corr, str):
                a_corr = [self.to_single_token(a_corr)]
            
            return logits[0, -1, a_clean] - logits[0, -1, a_corr].mean(-1)
        else:
            return logits[0, -1, a_clean]

    def plot_atp(self, atp, x_clean, component, mask=None, n_last_tokens=128, val=1, prepend_bos=True):
        str_tokens = self.to_str_tokens(x_clean, prepend_bos=prepend_bos)
        xs = [f"{tok} | {i}" for i, tok in enumerate(str_tokens[-n_last_tokens:])]
        
        if component in ['z', 'q', 'result']:
            ys = [f'L{i}H{j}' for i in range(self.cfg.n_layers) for j in range(self.cfg.n_heads)]
        elif component in ['k', 'v']:
            ys = [f'L{i}{component.upper()}{j}' for i in range(self.cfg.n_layers) for j in range(self.cfg.n_key_value_heads)]
        else:
            ys = [f"L{l} {component.upper()}" for l in range(self.cfg.n_layers)]
        
        if mask is None:
            mask = torch.ones(atp.shape[0], dtype=bool)
        fig = px.imshow(
            atp[mask, -n_last_tokens:].cpu().numpy(), 
            x=xs,
            y=np.array(ys)[mask],
            color_continuous_scale='RdBu', zmin=-val, zmax=val, aspect='auto'
        )
        
        return fig