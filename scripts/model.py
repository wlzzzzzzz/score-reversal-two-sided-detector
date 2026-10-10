# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# third_party/fast-detect-gpt-LICENSE file.
# Modified: retain paper proxy loading and remove unused model/demo branches.

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import os

def from_pretrained(cls, model_name, kwargs, cache_dir, **extra_kwargs):
    # use local model if it exists
    kwargs = {**kwargs, **extra_kwargs}
    local_path = os.path.join(cache_dir, 'local.' + model_name.replace("/", "_"))
    if os.path.exists(local_path):
        return cls.from_pretrained(local_path, **kwargs)
    return cls.from_pretrained(model_name, **kwargs, cache_dir=cache_dir)

# Paper proxy aliases; explicit local paths and model identifiers also work.
model_fullnames = {
    'opt-2.7b': 'facebook/opt-2.7b',
    'falcon-7b': 'tiiuae/falcon-7b',
    'falcon-7b-instruct': 'tiiuae/falcon-7b-instruct',
}
float16_models = ['falcon-7b', 'falcon-7b-instruct']

def get_model_fullname(model_name):
    return model_fullnames[model_name] if model_name in model_fullnames else model_name

def get_tail_name(model_name):
    return os.path.basename(str(model_name).rstrip('/'))

def load_model(model_name, device, cache_dir):
    model_fullname = get_model_fullname(model_name)
    model_tail_name = get_tail_name(model_name)
    print(f'Loading model {model_fullname}...')
    model_kwargs = {}
    if str(device).startswith('cuda') or model_name in float16_models:
        model_kwargs.update(dict(torch_dtype=torch.float16))
    trust_remote_code = True
    if "falcon" in model_tail_name.lower():
        trust_remote_code = False
    if os.path.isdir(model_fullname):
        model_kwargs['local_files_only'] = True
    model = from_pretrained(
        AutoModelForCausalLM, model_fullname, model_kwargs, cache_dir,
        trust_remote_code=trust_remote_code,
    )
    print('Moving model to GPU...', end='', flush=True)
    start = time.time()
    model.to(device)
    print(f'DONE ({time.time() - start:.2f}s)')
    return model

def load_tokenizer(model_name, cache_dir):
    model_fullname = get_model_fullname(model_name)
    optional_tok_kwargs = {}
    model_tail_name = get_tail_name(model_name)
    trust_remote_code = True
    if "falcon" in model_tail_name.lower():
        trust_remote_code = False
    if "facebook/opt-" in model_fullname:
        print("Using non-fast tokenizer for OPT")
        optional_tok_kwargs['use_fast'] = False
    optional_tok_kwargs['padding_side'] = 'right'
    if os.path.isdir(model_fullname):
        optional_tok_kwargs['local_files_only'] = True
    base_tokenizer = from_pretrained(
        AutoTokenizer,
        model_fullname,
        optional_tok_kwargs,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
    )
    if base_tokenizer.pad_token_id is None:
        base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
    return base_tokenizer
