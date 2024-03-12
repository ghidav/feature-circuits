from collections import namedtuple
import torch as t
from tqdm import tqdm
from tkdict import TKDict
from numpy import ndindex
from typing import Dict, Union
from activation_utils import SparseAct

EffectOut = namedtuple('EffectOut', ['effects', 'deltas', 'grads', 'total_effect'])

def _pe_attrib_all_folded_sparseact(
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
    with model.trace(clean):
        for submodule in submodules:
            dictionary = dictionaries[submodule]
            x = submodule.output
            is_resid = (type(x.shape) == tuple)
            if is_resid:
                x = x[0]
            x_hat, f = dictionary(x, output_features=True) # x_hat implicitly depends on f
            residual = x - x_hat
            hidden_states_clean[submodule] = SparseAct(act=f, res=residual).save()
            grads[submodule] = hidden_states_clean[submodule].grad.save()
            residual.grad = t.zeros_like(residual)
            x_recon = x_hat + residual
            if is_resid:
                submodule.output[0][:] = x_recon
            else:
                submodule.output = x_recon
            x.grad = x_recon.grad
        metric_clean = metric_fn(model, **metric_kwargs).save()
        metric_clean.sum().backward()
    hidden_states_clean = {k : v.value for k, v in hidden_states_clean.items()}
    grads = {k : v.value for k, v in grads.items()}

    if patch is None:
        hidden_states_patch = {
            k : SparseAct(act=t.zeros_like(v.act), res=t.zeros_like(v.res)) for k, v in hidden_states_clean.items()
        }
        total_effect = None
    else:
        hidden_states_patch = {}
        with model.trace(patch), t.inference_mode():
            for submodule in submodules:
                dictionary = dictionaries[submodule]
                x = submodule.output
                is_resid = (type(x.shape) == tuple)
                if is_resid:
                    x = x[0]
                x_hat, f = dictionary(x, output_features=True)
                residual = x - x_hat
                hidden_states_patch[submodule] = SparseAct(act=f, res=residual).save()
            metric_patch = metric_fn(model, **metric_kwargs).save()
        total_effect = (metric_patch.value - metric_clean.value).detach()
        hidden_states_patch = {k : v.value for k, v in hidden_states_patch.items()}

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
    
    return EffectOut(effects, deltas, grads, total_effect)

def _pe_ig_sparseact(
        clean,
        patch,
        model,
        submodules,
        dictionaries,
        metric_fn,
        steps=10,
        metric_kwargs=dict(),
):

    hidden_states_clean = {}
    is_resids = {}
    with model.trace(clean), t.inference_mode():
        for submodule in submodules:
            dictionary = dictionaries[submodule]
            x = submodule.output
            is_resids[submodule] = (type(x.shape) == tuple)
            if is_resids[submodule]:
                x = x[0]
            f = dictionary.encode(x)
            x_hat = dictionary.decode(f)
            residual = x - x_hat
            hidden_states_clean[submodule] = SparseAct(act=f.save(), res=residual.save())
        metric_clean = metric_fn(model, **metric_kwargs).save()
    hidden_states_clean = {k : v.value for k, v in hidden_states_clean.items()}

    if patch is None:
        hidden_states_patch = {
            k : SparseAct(act=t.zeros_like(v.act), res=t.zeros_like(v.res)) for k, v in hidden_states_clean.items()
        }
        total_effect = None
    else:
        hidden_states_patch = {}
        with model.trace(patch), t.inference_mode():
            for submodule in submodules:
                dictionary = dictionaries[submodule]
                x = submodule.output
                if is_resids[submodule]:
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
        with model.trace() as tracer:
            metrics = []
            fs = []
            for step in range(steps):
                alpha = step / steps
                f = (1 - alpha) * clean_state + alpha * patch_state
                f.act.retain_grad()
                f.res.retain_grad()
                fs.append(f)
                with tracer.invoke(clean):
                    if is_resids[submodule]:
                        submodule.output[0][:] = dictionary.decode(f.act) + f.res # clean_state.res instead of f.res makes this exactly same as the non-sparseact version
                    else:
                        submodule.output = dictionary.decode(f.act) + f.res # clean_state.res instead of f.res makes this exactly same as the non-sparseact version
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
    total_effect = total_effect if total_effect is not None else None

    return EffectOut(effects, deltas, grads, total_effect)


def _pe_exact_sparseact(
    clean,
    patch,
    model,
    submodules,
    dictionaries,
    metric_fn,
    ):
    hidden_states_clean = {}
    is_resids = {}
    with model.trace(clean), t.inference_mode():
        for submodule in submodules:
            dictionary = dictionaries[submodule]
            x = submodule.output
            is_resids[submodule] = (type(x.shape) == tuple)
            if is_resids[submodule]:
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
        with model.trace(patch), t.inference_mode():
            for submodule in submodules:
                dictionary = dictionaries[submodule]
                x = submodule.output
                if is_resids[submodule]:
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
        effect = SparseAct(act=t.zeros_like(clean_state.act), res=t.zeros_like(clean_state.res))
        
        # iterate over positions and features for which clean and patch differ
        idxs = t.nonzero(patch_state.act - clean_state.act)
        for idx in tqdm(idxs):
            with model.trace(clean), t.inference_mode():
                f = clean_state.act.clone()
                f[tuple(idx)] = patch_state.act[tuple(idx)]
                x_hat = dictionary.decode(f)
                if is_resids[submodule]:
                    submodule.output[0][:] = x_hat + clean_state.res
                else:
                    submodule.output = x_hat + clean_state.res
                metric = metric_fn(model).save()
            effect.act[tuple(idx)] = (metric.value - metric_clean.value).sum()
        for idx in list(ndindex(*clean_state.res.shape[:-1])):
            with model.trace(clean), t.inference_mode():
                res = clean_state.res.clone()
                res[tuple(idx)] = patch_state.res[tuple(idx)]
                x_hat = dictionary.decode(clean_state.act)
                if is_resids[submodule]:
                    submodule.output[0][:] = x_hat + res
                else:
                    submodule.output = x_hat + res
                metric = metric_fn(model).save()
            effect.res[tuple(idx)] = (metric.value - metric_clean.value).sum()
        
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
        method='all-folded',
        steps=10,
        metric_kwargs=dict()
):
    if method == 'all-folded':
        return _pe_attrib_all_folded_sparseact(clean, patch, model, submodules, dictionaries, metric_fn, metric_kwargs=metric_kwargs)
    elif method == 'ig':
        return _pe_ig_sparseact(clean, patch, model, submodules, dictionaries, metric_fn, steps=steps, metric_kwargs=metric_kwargs)
    elif method == 'exact':
        return _pe_exact_sparseact(clean, patch, model, submodules, dictionaries, metric_fn)
    else:
        raise ValueError(f"Unknown method {method}")

def jvp(
        input,
        model,
        dictionaries,
        downstream_submod,
        downstream_features,
        upstream_submod,
        left_vec : Union[SparseAct, Dict[int, SparseAct]],
        right_vec : SparseAct,
        return_without_right = False,
):
    """
    Return a sparse shape [# downstream features + 1, # upstream features + 1] tensor of Jacobian-vector products.
    """

    if not downstream_features: # handle empty list
        if not return_without_right:
            return t.sparse_coo_tensor(t.zeros((2, 0), dtype=t.long), t.zeros(0)).to(model.device)
        else:
            return t.sparse_coo_tensor(t.zeros((2, 0), dtype=t.long), t.zeros(0)).to(model.device), t.sparse_coo_tensor(t.zeros((2, 0), dtype=t.long), t.zeros(0)).to(model.device)

    downstream_dict, upstream_dict = dictionaries[downstream_submod], dictionaries[upstream_submod]

    vjv_indices = {}
    vjv_values = {}
    if return_without_right:
        jv_indices = {}
        jv_values = {}

    with model.trace(input):
        # first specify forward pass modifications
        x = upstream_submod.output
        is_resid = (type(x.shape) == tuple)
        if is_resid:
            x = x[0]
        x_hat, f = upstream_dict(x, output_features=True)
        x_res = x - x_hat
        upstream_act = SparseAct(act=f, res=x_res).save()
        if is_resid:
            upstream_submod.output[0][:] = x_hat + x_res
        else:
            upstream_submod.output = x_hat + x_res
        y = downstream_submod.output
        if type(y.shape) == tuple:
            y = y[0]
        y_hat, g = downstream_dict(y, output_features=True)
        y_res = y - y_hat
        downstream_act = SparseAct(act=g, res=y_res).save()

        for downstream_feat in downstream_features:
            if isinstance(left_vec, SparseAct):
                to_backprop = (left_vec @ downstream_act).to_tensor().flatten()
            elif isinstance(left_vec, dict):
                to_backprop = (left_vec[downstream_feat] @ downstream_act).to_tensor().flatten()
            else:
                raise ValueError(f"Unknown type {type(left_vec)}")
            vjv = (upstream_act.grad @ right_vec).to_tensor().flatten()
            if return_without_right:
                jv = (upstream_act.grad @ right_vec).to_tensor().flatten()
            x_res.grad = t.zeros_like(x_res)
            to_backprop[downstream_feat].backward(retain_graph=True)

            vjv_indices[downstream_feat] = vjv.nonzero().squeeze(-1).save()
            vjv_values[downstream_feat] = vjv[vjv_indices[downstream_feat]].save()

            if return_without_right:
                jv_indices[downstream_feat] = jv.nonzero().squeeze(-1).save()
                jv_values[downstream_feat] = jv[vjv_indices[downstream_feat]].save()

    # get shapes
    d_downstream_contracted = len((downstream_act.value @ downstream_act.value).to_tensor().flatten())
    d_upstream_contracted = len((upstream_act.value @ upstream_act.value).to_tensor().flatten())
    if return_without_right:
        d_upstream = len(upstream_act.value.to_tensor().flatten())


    vjv_indices = t.tensor(
        [[downstream_feat for downstream_feat in downstream_features for _ in vjv_indices[downstream_feat].value],
         t.cat([vjv_indices[downstream_feat].value for downstream_feat in downstream_features], dim=0)]
    ).to(model.device)
    vjv_values = t.cat([vjv_values[downstream_feat].value for downstream_feat in downstream_features], dim=0)

    if not return_without_right:
        return t.sparse_coo_tensor(vjv_indices, vjv_values, (d_downstream_contracted, d_upstream_contracted))

    jv_indices = t.tensor(
        [[downstream_feat for downstream_feat in downstream_features for _ in jv_indices[downstream_feat].value],
         t.cat([jv_indices[downstream_feat].value for downstream_feat in downstream_features], dim=0)]
    ).to(model.device)
    jv_values = t.cat([jv_values[downstream_feat].value for downstream_feat in downstream_features], dim=0)

    return (
        t.sparse_coo_tensor(vjv_indices, vjv_values, (d_downstream_contracted, d_upstream_contracted)),
        t.sparse_coo_tensor(jv_indices, jv_values, (d_downstream_contracted, d_upstream))
    )


if __name__ == "__main__":
    from nnsight import LanguageModel
    from dictionary_learning import AutoEncoder

    model = LanguageModel('EleutherAI/pythia-70m-deduped', device_map='cpu')
    submodules = []
    submodule_names = {}
    dictionaries = {}
    for layer in range(len(model.gpt_neox.layers)):
        submodule = model.gpt_neox.layers[layer].mlp
        submodule_names[submodule] = f'mlp{layer}'
        submodules.append(submodule)
        ae = AutoEncoder(512, 64 * 512)#.cuda()
        ae.load_state_dict(t.load(f'/share/projects/dictionary_circuits/autoencoders/pythia-70m-deduped/mlp_out_layer{layer}/5_32768/ae.pt'))
        dictionaries[submodule] = ae

        submodule = model.gpt_neox.layers[layer]
        submodule_names[submodule] = f'resid{layer}'
        submodules.append(submodule)
        ae = AutoEncoder(512, 64 * 512)#.cuda()
        ae.load_state_dict(t.load(f'/share/projects/dictionary_circuits/autoencoders/pythia-70m-deduped/resid_out_layer{layer}/5_32768/ae.pt'))
        dictionaries[submodule] = ae

    clean_context = ["The man"] #, "The tall boy"]
    patch_context = ["The men"] #, "The tall boys"]
    clean_idx = model.tokenizer(" is").input_ids[-1]
    patch_idx = model.tokenizer(" are").input_ids[-1]

    def metric_fn(model):
        return model.embed_out.output[:,-1,patch_idx] - model.embed_out.output[:,-1,clean_idx]
    
    def compare_effect_outs(eo1, eo2_sparseact):
        for k in ['effects', 'deltas', 'grads']:
            effect_out1 = getattr(eo1, k)
            if effect_out1 is None:
                continue
            for submod in effect_out1:
                tensor1 = effect_out1[submod]
                effect_out2_sparseact = getattr(eo2_sparseact, k)[submod]
                if isinstance(effect_out2_sparseact, SparseAct):
                    tensor2_sparseact = effect_out2_sparseact.act
                else:
                    tensor2_sparseact = effect_out2_sparseact
                if not t.allclose(tensor1, tensor2_sparseact):
                    print(f"{k} differs at submod {submod}")
                    print(tensor1.sum())
                    print(tensor2_sparseact.sum())
                    return False
        return True

    # Check that the sparseact version of the function returns the same result as the original

    # ## All-folded feature activation test
    # effect_out_all_folded = _pe_attrib_all_folded(
    #     clean_context,
    #     patch_context,
    #     model,
    #     submodules,
    #     dictionaries,
    #     metric_fn,
    # )
    # effect_out_all_folded_sparseact = _pe_attrib_all_folded_sparseact(
    #     clean_context,
    #     patch_context,
    #     model,
    #     submodules,
    #     dictionaries,
    #     metric_fn,
    # )
    # if compare_effect_outs(effect_out_all_folded, effect_out_all_folded_sparseact):
    #   print("All-folded test passed")

    # ## IG feature activation test
    # effect_out_ig = _pe_ig(
    #     clean_context,
    #     patch_context,
    #     model,
    #     submodules,
    #     dictionaries,
    #     metric_fn,
    # )
    # effect_out_ig_sparseact = _pe_ig_sparseact(
    #     clean_context,
    #     patch_context,
    #     model,
    #     submodules,
    #     dictionaries,
    #     metric_fn,
    # )
    # if compare_effect_outs(effect_out_ig, effect_out_ig_sparseact):
    #     print("IG test passed")

    ## Exact feature activation test
    effect_out_exact = _pe_exact(
        clean_context,
        patch_context,
        model,
        submodules,
        dictionaries,
        metric_fn,
    )
    effect_out_exact_sparseact = _pe_exact_sparseact(
        clean_context,
        patch_context,
        model,
        submodules,
        dictionaries,
        metric_fn,
    )
    if compare_effect_outs(effect_out_exact, effect_out_exact_sparseact):
        print("Exact test passed")