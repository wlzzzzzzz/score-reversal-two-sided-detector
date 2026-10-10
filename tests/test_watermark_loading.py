"""Configuration loading for the seven retained paper watermark algorithms."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/MarkLLM"))

from watermark.auto_config import AutoConfig
from watermark.auto_watermark import AutoWatermark
from watermark.kgw import KGWConfig
from utils.utils import load_config_file


def configuration():
    return SimpleNamespace(model=None, tokenizer=None, vocab_size=64,
                           device="cpu", gen_kwargs={})


@pytest.mark.parametrize("name", ["KGW", "SWEET", "Unigram", "EXP", "SynthID", "DIP", "ITSEdit"])
def test_bundled_configuration_loads_outside_repository(name, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = AutoConfig.load(name, configuration())
    assert config.algorithm_name == name
    assert config.config_dict["algorithm_name"] == name


def test_base_config_default_and_hash_override(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = KGWConfig(None, configuration())
    assert config.hash_key == 15485863
    watermark = AutoWatermark.load("KGW", transformers_config=configuration(), hash_key="123")
    assert watermark.config.hash_key == 123
    assert watermark.utils.config is watermark.config


def test_configuration_errors_keep_their_original_cause(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config_file(tmp_path / "missing.json")
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{")
    with pytest.raises(json.JSONDecodeError):
        load_config_file(invalid)
