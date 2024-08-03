from collections import namedtuple
import torch as t
from tqdm import tqdm
from numpy import ndindex
from typing import Dict, Union
from activation_utils import SparseAct
from functools import partial
import gc

DEBUGGING = False

if DEBUGGING:
    tracer_kwargs = {'validate' : True, 'scan' : True}
else:
    tracer_kwargs = {'validate' : False, 'scan' : False}

EffectOut = namedtuple('EffectOut', ['effects', 'deltas', 'grads', 'total_effect'])

def _pe_attrib(
        clean,
        patch,
        model,
        submodules,
        dictionaries,
        metric_fn,
        metric_kwargs=dict(),
):
    
    hidden_states_clean = {}
    grads = {}

    sae_hooks = []

    def sae_hook(act, hook, sae, cache):
        original_shape = act.shape
        
        if len(original_shape) == 4:
            x = act.reshape(act.shape[0], act.shape[1], -1).clone()
        else:
            x = act.clone()

        x.requires_grad_(True)
        
        x_hat, f = sae(x, output_features=True)
    
        residual = x - x_hat

        if t.is_grad_enabled():
            f.retain_grad()

        cache[hook.name] = SparseAct(act=f, res=residual)

        x_recon = x_hat + residual.detach()
        
        if len(original_shape) == 4:
            return x_recon.reshape(original_shape)

        return x_recon

    for i, submodule in enumerate(submodules):
        dictionary = dictionaries[submodule]
        sae_hooks.append((submodule, partial(sae_hook, sae=dictionary, cache=hidden_states_clean)))

    # Forward pass with hooks
    logits = model.run_with_hooks(clean, fwd_hooks=sae_hooks)
    metric_clean = metric_fn(logits, **metric_kwargs)

    # Backward pass
    metric_clean.sum().backward()

    # Collect gradients
    for submodule in submodules:
        if submodule in hidden_states_clean:
            grads[submodule] = hidden_states_clean[submodule].grad
            grads[submodule].res = t.zeros_like(hidden_states_clean[submodule].res)

    #hidden_states_clean = {k : v.value for k, v in hidden_states_clean.items()}
    #grads = {k : v.value for k, v in grads.items()}

    if patch is None:
        hidden_states_patch = {
            k : SparseAct(act=t.zeros_like(v.act), res=t.zeros_like(v.res)) for k, v in hidden_states_clean.items()
        }
        total_effect = None
    else:
        hidden_states_patch = {}
        sae_hooks = []
        for i, submodule in enumerate(submodules):
            dictionary = dictionaries[submodule]
            sae_hooks.append((submodule, partial(sae_hook, sae=dictionary, cache=hidden_states_patch)))
        
        with t.no_grad():
            corr_logits = model.run_with_hooks(patch, fwd_hooks=sae_hooks)
        metric_patch = metric_fn(corr_logits, **metric_kwargs)
        total_effect = (metric_patch - metric_clean).detach()
        #hidden_states_patch = {k : v.value for k, v in hidden_states_patch.items()}

    effects = {}
    deltas = {}
    for submodule in submodules:
        patch_state, clean_state, grad = hidden_states_patch[submodule], hidden_states_clean[submodule], grads[submodule]
        delta = patch_state - clean_state.detach() if patch_state is not None else -clean_state.detach()
        effect = delta @ grad
        effects[submodule] = effect
        deltas[submodule] = delta
        grads[submodule] = grad
    total_effect = total_effect if total_effect is not None else None

    del hidden_states_clean, hidden_states_patch
    gc.collect()
    
    return EffectOut(effects, deltas, grads, total_effect)

def _pe_ig(
        clean,
        patch,
        model,
        submodules,
        dictionaries,
        metric_fn,
        steps=10,
        metric_kwargs=dict(),
):
    
    # first run through a test input to figure out which hidden states are tuples
    is_tuple = {}
    with model.trace("_"):
        for submodule in submodules:
            is_tuple[submodule] = type(submodule.output.shape) == tuple

    hidden_states_clean = {}
    with model.trace(clean, **tracer_kwargs), t.no_grad():
        for submodule in submodules:
            dictionary = dictionaries[submodule]
            x = submodule.output
            if is_tuple[submodule]:
                x = x[0]
            f = dictionary.encode(x)
            x_hat = dictionary.decode(f)
            residual = x - x_hat
            hidden_states_clean[submodule] = SparseAct(act=f.save(), res=residual.save())
        metric_clean = metric_fn(model, **metric_kwargs).save()
    hidden_states_clean = {k : v for k, v in hidden_states_clean.items()}

    if patch is None:
        hidden_states_patch = {
            k : SparseAct(act=t.zeros_like(v.act), res=t.zeros_like(v.res)) for k, v in hidden_states_clean.items()
        }
        total_effect = None
    else:
        hidden_states_patch = {}
        with model.trace(patch, **tracer_kwargs), t.no_grad():
            for submodule in submodules:
                dictionary = dictionaries[submodule]
                x = submodule.output
                if is_tuple[submodule]:
                    x = x[0]
                f = dictionary.encode(x)
                x_hat = dictionary.decode(f)
                residual = x - x_hat
                hidden_states_patch[submodule] = SparseAct(act=f.save(), res=residual.save())
            metric_patch = metric_fn(model, **metric_kwargs).save()
        total_effect = (metric_patch.value - metric_clean.value).detach()
        hidden_states_patch = {k : v.value for k, v in hidden_states_patch.items()}

    effects = {}
    deltas = {}
    grads = {}
    for submodule in submodules:
        dictionary = dictionaries[submodule]
        clean_state = hidden_states_clean[submodule]
        patch_state = hidden_states_patch[submodule]
        with model.trace(**tracer_kwargs) as tracer:
            metrics = []
            fs = []
            for step in range(steps):
                alpha = step / steps
                f = (1 - alpha) * clean_state + alpha * patch_state
                f.act.retain_grad()
                f.res.retain_grad()
                fs.append(f)
                with tracer.invoke(clean, scan=tracer_kwargs['scan']):
                    if is_tuple[submodule]:
                        submodule.output[0][:] = dictionary.decode(f.act) + f.res
                    else:
                        submodule.output = dictionary.decode(f.act) + f.res
                    metrics.append(metric_fn(model, **metric_kwargs))
            metric = sum([m for m in metrics])
            metric.sum().backward(retain_graph=True) # TODO : why is this necessary? Probably shouldn't be, contact jaden

        mean_grad = sum([f.act.grad for f in fs]) / steps
        mean_residual_grad = sum([f.res.grad for f in fs]) / steps
        grad = SparseAct(act=mean_grad, res=mean_residual_grad)
        delta = (patch_state - clean_state).detach() if patch_state is not None else -clean_state.detach()
        effect = grad @ delta

        effects[submodule] = effect
        deltas[submodule] = delta
        grads[submodule] = grad

    return EffectOut(effects, deltas, grads, total_effect)


def _pe_exact(
    clean,
    patch,
    model,
    submodules,
    dictionaries,
    metric_fn,
    ):

    # first run through a test input to figure out which hidden states are tuples
    is_tuple = {}
    with model.trace("_"):
        for submodule in submodules:
            is_tuple[submodule] = type(submodule.output.shape) == tuple

    hidden_states_clean = {}
    with model.trace(clean, **tracer_kwargs), t.inference_mode():
        for submodule in submodules:
            dictionary = dictionaries[submodule]
            x = submodule.output
            if is_tuple[submodule]:
                x = x[0]
            f = dictionary.encode(x)
            x_hat = dictionary.decode(f)
            residual = x - x_hat
            hidden_states_clean[submodule] = SparseAct(act=f, res=residual).save()
        metric_clean = metric_fn(model).save()
    hidden_states_clean = {k : v.value for k, v in hidden_states_clean.items()}

    if patch is None:
        hidden_states_patch = {
            k : SparseAct(act=t.zeros_like(v.act), res=t.zeros_like(v.res)) for k, v in hidden_states_clean.items()
        }
        total_effect = None
    else:
        hidden_states_patch = {}
        with model.trace(patch, **tracer_kwargs), t.inference_mode():
            for submodule in submodules:
                dictionary = dictionaries[submodule]
                x = submodule.output
                if is_tuple[submodule]:
                    x = x[0]
                f = dictionary.encode(x)
                x_hat = dictionary.decode(f)
                residual = x - x_hat
                hidden_states_patch[submodule] = SparseAct(act=f, res=residual).save()
            metric_patch = metric_fn(model).save()
        total_effect = metric_patch.value - metric_clean.value
        hidden_states_patch = {k : v.value for k, v in hidden_states_patch.items()}

    effects = {}
    deltas = {}
    for submodule in submodules:
        dictionary = dictionaries[submodule]
        clean_state = hidden_states_clean[submodule]
        patch_state = hidden_states_patch[submodule]
        effect = SparseAct(act=t.zeros_like(clean_state.act), resc=t.zeros(*clean_state.res.shape[:-1])).to(device)
        
        # iterate over positions and features for which clean and patch differ
        idxs = t.nonzero(patch_state.act - clean_state.act)
        for idx in tqdm(idxs):
            with t.inference_mode():
                with model.trace(clean, **tracer_kwargs):
                    f = clean_state.act.clone()
                    f[tuple(idx)] = patch_state.act[tuple(idx)]
                    x_hat = dictionary.decode(f)
                    if is_tuple[submodule]:
                        submodule.output[0][:] = x_hat + clean_state.res
                    else:
                        submodule.output = x_hat + clean_state.res
                    metric = metric_fn(model).save()
                effect.act[tuple(idx)] = (metric.value - metric_clean.value).sum()

        for idx in list(ndindex(effect.resc.shape)):
            with t.inference_mode():
                with model.trace(clean, **tracer_kwargs):
                    res = clean_state.res.clone()
                    res[tuple(idx)] = patch_state.res[tuple(idx)]
                    x_hat = dictionary.decode(clean_state.act)
                    if is_tuple[submodule]:
                        submodule.output[0][:] = x_hat + res
                    else:
                        submodule.output = x_hat + res
                    metric = metric_fn(model).save()
                effect.resc[tuple(idx)] = (metric.value - metric_clean.value).sum()
        
        effects[submodule] = effect
        deltas[submodule] = patch_state - clean_state
    total_effect = total_effect if total_effect is not None else None

    return EffectOut(effects, deltas, None, total_effect)

def patching_effect(
        clean,
        patch,
        model,
        submodules,
        dictionaries,
        metric_fn,
        method='attrib',
        steps=10,
        metric_kwargs=dict()
):
    if method == 'attrib':
        return _pe_attrib(clean, patch, model, submodules, dictionaries, metric_fn, metric_kwargs=metric_kwargs)
    elif method == 'ig':
        return _pe_ig(clean, patch, model, submodules, dictionaries, metric_fn, steps=steps, metric_kwargs=metric_kwargs)
    elif method == 'exact':
        return _pe_exact(clean, patch, model, submodules, dictionaries, metric_fn)
    else:
        raise ValueError(f"Unknown method {method}")

def jvp(
        input,
        model,
        dictionaries,
        downstream_submod,
        downstream_features,
        upstream_submod,
        left_vec,
        right_vec,
        return_without_right = False,
        device='cuda'
):
    """
    Return a sparse shape [# downstream features + 1, # upstream features + 1] tensor of Jacobian-vector products.
    """

    if not downstream_features: # handle empty list
        if not return_without_right:
            return t.sparse_coo_tensor(t.zeros((2, 0), dtype=t.long), t.zeros(0)).to(device)
        else:
            return t.sparse_coo_tensor(t.zeros((2, 0), dtype=t.long), t.zeros(0)).to(device), t.sparse_coo_tensor(t.zeros((2, 0), dtype=t.long), t.zeros(0)).to(device)

    downstream_dict, upstream_dict = dictionaries[downstream_submod], dictionaries[upstream_submod]

    vjv_indices = {}
    vjv_values = {}
    
    if return_without_right:
        jv_indices = {}
        jv_values = {}

    cache = {}

    def sae_hook(act, hook, sae, cache):
        original_shape = act.shape
        
        if len(original_shape) == 4:
            x = act.reshape(act.shape[0], act.shape[1], -1).clone()
        else:
            x = act.clone()
        
        x_hat, f = sae(x, output_features=True)
        
        if t.is_grad_enabled():
            f.retain_grad()
        
        residual = x - x_hat
        cache[hook.name] = SparseAct(act=f, res=residual)

        #residual.grad = t.zeros_like(residual)

        x_recon = x_hat + residual.detach()
        
        if len(original_shape) == 4:
            return x_recon.reshape(original_shape)

        return x_recon

    sae_hook = [
        (upstream_submod, partial(sae_hook, sae=upstream_dict, cache=cache)),
        (downstream_submod, partial(sae_hook, sae=downstream_dict, cache=cache))
    ]

    _ = model.run_with_hooks(input, fwd_hooks=sae_hook)

    downstream_act = cache[downstream_submod]
    upstream_act = cache[upstream_submod]
    
    for downstream_feat in downstream_features:
            if isinstance(left_vec, SparseAct):
                to_backprop = (left_vec @ downstream_act).to_tensor().flatten()
            elif isinstance(left_vec, dict):
                to_backprop = (left_vec[downstream_feat] @ downstream_act).to_tensor().flatten()
            else:
                raise ValueError(f"Unknown type {type(left_vec)}")

            to_backprop[downstream_feat].backward(retain_graph=True)

            upstream_act_grad = upstream_act.grad
            downstream_act_grad = downstream_act.grad

            upstream_act_grad.res = t.zeros_like(upstream_act.res)
            downstream_act_grad.res = t.zeros_like(downstream_act.res)

            vjv = (upstream_act_grad @ right_vec).to_tensor().flatten()
            if return_without_right:
                jv = (upstream_act_grad @ right_vec).to_tensor().flatten()
            #x_res.grad = t.zeros_like(x_res)
            
            vjv_indices[downstream_feat] = vjv.nonzero().squeeze(-1)
            vjv_values[downstream_feat] = vjv[vjv_indices[downstream_feat]]

            if return_without_right:
                jv_indices[downstream_feat] = jv.nonzero().squeeze(-1)
                jv_values[downstream_feat] = jv[vjv_indices[downstream_feat]]

    # get shapes
    d_downstream_contracted = len((downstream_act @ downstream_act).to_tensor().flatten())
    d_upstream_contracted = len((upstream_act @ upstream_act).to_tensor().flatten())
    if return_without_right:
        d_upstream = len(upstream_act.to_tensor().flatten())


    vjv_indices = t.tensor(
        [[downstream_feat for downstream_feat in downstream_features for _ in vjv_indices[downstream_feat]],
         t.cat([vjv_indices[downstream_feat] for downstream_feat in downstream_features], dim=0)]
    ).to(device)
    vjv_values = t.cat([vjv_values[downstream_feat] for downstream_feat in downstream_features], dim=0)

    if not return_without_right:
        return t.sparse_coo_tensor(vjv_indices, vjv_values, (d_downstream_contracted, d_upstream_contracted))

    jv_indices = t.tensor(
        [[downstream_feat for downstream_feat in downstream_features for _ in jv_indices[downstream_feat]],
         t.cat([jv_indices[downstream_feat] for downstream_feat in downstream_features], dim=0)]
    ).to(device)
    jv_values = t.cat([jv_values[downstream_feat] for downstream_feat in downstream_features], dim=0)

    return (
        t.sparse_coo_tensor(vjv_indices, vjv_values, (d_downstream_contracted, d_upstream_contracted)),
        t.sparse_coo_tensor(jv_indices, jv_values, (d_downstream_contracted, d_upstream))
    )