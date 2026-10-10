"""Deterministic seeds and model loading for the pinned generation environment."""
import hashlib
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def stable_seed(base_seed: int, *parts: object) -> int:
    material = "\x1f".join([str(base_seed), *[str(part) for part in parts]])
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:4], "big")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(model_path: Path, max_new_tokens: int, native_falcon: bool = False):
    """Load local checkpoints, using native BF16 Falcon when requested."""
    trust_remote_code = not native_falcon
    config = AutoConfig.from_pretrained(
        model_path, trust_remote_code=trust_remote_code, local_files_only=True,
    )
    added_max_length = False
    if not hasattr(config, "max_length"):
        config.max_length = max_new_tokens
        added_max_length = True
    if not hasattr(config, "use_cache"):
        config.use_cache = True
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        device_map="auto",
        torch_dtype=(torch.bfloat16 if native_falcon else torch.float16),
        trust_remote_code=trust_remote_code,
        local_files_only=True,
    ).eval()
    if added_max_length and hasattr(model.config, "max_length"):
        delattr(model.config, "max_length")
    return model, tokenizer
