import json
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
import os

BASE_DIR = "/workspace/feature-circuits"

with open(os.path.join(BASE_DIR, "keys.json"), "r") as f:
    keys = json.load(f)

def generate_dataset(N, M=8, E=0, zeros=0, complete=False):

    def add_zeros(x, L):
        return "0" * max(0, zeros - len(str(x))) + str(x)

    examples = {
        "prompt": [],
        "a": [],
        "b": [],
        "solution": [],
    }

    for n in range(0, N):
        for _ in range(M):

            prompt = ""

            for _ in range(E):
                a, b = np.random.randint(0, N // 2, 2)
                c = add_zeros(a + b, zeros)
                a, b = add_zeros(a, zeros), add_zeros(b, zeros)
                prompt += f"{a}+{b}={c}\n"

            if n == 0:
                c = 0
            else:
                c = np.random.randint(0, n)

            prompt += f"{add_zeros(c, zeros)}+{add_zeros(n - c, zeros)}="
            
            if complete:
                prompt += f"{add_zeros(n, zeros)}"

            examples["prompt"].append(prompt)
            examples["a"].append(c)
            examples["b"].append(n - c)
            examples["solution"].append(n)

    examples = pd.DataFrame(examples)

    def carry_u(x):
        return (int(str(x["a"])[-1]) + int(str(x["b"])[-1])) > 9

    def carry_d(x):
        d_a = 0 if len(str(x["a"])) < 2 else int(str(x["a"])[-2])
        d_b = 0 if len(str(x["b"])) < 2 else int(str(x["b"])[-2])
        return (d_a + d_b + int(x["carry_u"])) > 9

    def carry_h(x):
        h_a = 0 if len(str(x["a"])) < 3 else int(str(x["a"])[-3])
        h_b = 0 if len(str(x["b"])) < 3 else int(str(x["b"])[-3])
        return (h_a + h_b + int(x["carry_d"])) > 9

    examples["carry_u"] = examples.apply(carry_u, axis=1)
    examples["carry_d"] = examples.apply(carry_d, axis=1)

    return examples


def tensor_slice(tensor, start_idx=None, end_idx=None):
    if start_idx is not None:
        if end_idx is not None:
            return (
                tensor[:, :, start_idx:end_idx]
                if len(tensor.shape) == 4
                else tensor[:, start_idx:end_idx]
            )
        return (
            tensor[:, :, start_idx:]
            if len(tensor.shape) == 4
            else tensor[:, start_idx:]
        )
    elif end_idx is not None:
        return tensor[:, :, :end_idx] if len(tensor.shape) == 4 else tensor[:, :end_idx]


def get_activations(
    model, examples, component, bs=8, start_layer=0, start_idx=None, end_idx=None
):
    """
    Get activations for a given model and examples.
    model: The model to use for generation.
    examples: The examples dataframe to generate activations for.
    component: The component to extract activations from.
    bs: The batch size to use.
    start_layer: The layer to start extracting activations from.
    start_idx: The starting index for the activations.
    end_idx: The ending index for the activations.
    """
    activations = []

    for b in tqdm(range(0, len(examples), bs)):
        example = examples["prompt"].iloc[b : b + bs].tolist()
        tokens = model.to_tokens(example, prepend_bos=True, padding_side="left")
        with torch.no_grad():
            logits, cache = model.run_with_cache(tokens)

        if component == "ln_final":
            acts = cache["ln_final.hook_normalized"].cpu() # [bs, seq_len, dim]
        elif component == "logits":
            acts = logits.cpu() # [bs, seq_len, vocab_size]
        else:
            acts = cache.stack_activation(component)[start_layer:].cpu() # [layer, bs, seq_len, dim]

        activations.append(tensor_slice(acts, start_idx, end_idx))
        del cache

    activations = torch.cat(activations, dim=-3)
    return activations


def get_completions(model, examples, bs=32, K=3):
    """
    Generate completions for a given model and examples.
    model: The model to use for generation.
    examples: The examples dataframe to generate completions for.
    bs: The batch size to use.
    K: The number of tokens to generate.
    """
    completions = []

    for b in tqdm(range(0, len(examples), bs)):
        example = examples["prompt"].iloc[b : b + bs].tolist()
        tokens = model.to_tokens(
            example, prepend_bos=True, padding_side="left"
        )  # [bs, seq_len]

        for k in range(K):
            with torch.no_grad():
                logits = model(tokens)

            new_tok = logits[:, -1].argmax(-1)  # [bs]
            tokens = torch.cat([tokens, new_tok[:, None]], -1)

        completions += model.tokenizer.batch_decode(tokens, skip_special_tokens=True)

    completions = pd.Series(completions).apply(lambda x: x.strip())

    return completions


def get_predictions(completions):
    """
    Extract predictions from completions.
    completions: The completions to extract predictions from.
    """

    def extract_prediction(x):
        try:
            return int(x.split("=")[-1].split("\n")[0].split("$")[0].strip())
        except:
            return -1

    predictions = completions.apply(extract_prediction)

    return predictions