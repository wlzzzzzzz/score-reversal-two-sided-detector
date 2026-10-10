# Benchmark texts and source attribution

`benchmark/complete_prompt_free/<domain>/<generator>/data.json` contains the
80 frozen benchmark cells. The `original` list contains human continuations;
the eleven other lists contain the paper's machine-generation conditions.
Lists in a cell are aligned by index. File hashes and source counts are in
[`configs/benchmark_cells.json`](../configs/benchmark_cells.json).
The main evaluation excludes `wp:11`; the underlying text files retain it.

## Source routes

| Domain IDs | Human-source family | Route used by the preparation code |
| --- | --- | --- |
| `c4` | [C4](https://huggingface.co/datasets/allenai/c4) | Processed C4 input; deterministic replacement of damaged human rows |
| `squad` | [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) | Historical prompt/human source files |
| `writing` | [WritingPrompts](https://github.com/facebookresearch/fairseq/tree/main/examples/stories) | Historical prompt/human source files |
| `xsum` | [XSum](https://github.com/EdinburghNLP/XSum/tree/master/XSum-Dataset) | Historical prompt/human source files |
| `cmv`, `cnn`, `dialogsum`, `eli5`, `hswag`, `imdb`, `pubmed`, `roct`, `sci`, `tldr`, `wp`, `yelp` | [MAGE](https://github.com/yafuly/MAGE) human-source collections | Exact prompt-prefix matching against `train.csv`, `valid.csv`, `test.csv`, and `test_ood_set_gpt.csv` |

The recovery rules are implemented in `scripts/prepare_20cell_regeneration.py`
and `scripts/clean_clean41_text_separation.py`. Source-family links identify
upstream projects; they do not establish the exact upstream revision of every
historical input. `writing` and `wp` retain their distinct development/test
source identities in this benchmark.

Please credit the upstream dataset authors when reusing these texts. For
MAGE, see Li et al., *MAGE: Machine-generated Text Detection in the Wild*,
ACL 2024 ([paper and citation](https://aclanthology.org/2024.acl-long.3/)).
The other project pages above provide their source citations.

## Data terms and publication status

The repository's MIT license covers its original software and software
documentation. It does not assign a new license to source texts or model outputs.

- The [C4 card](https://huggingface.co/datasets/allenai/c4/raw/main/README.md)
  identifies ODC-BY; the underlying web documents can have separate rights.
- The [SQuAD provider](https://rajpurkar.github.io/SQuAD-explorer/)
  identifies CC BY-SA 4.0. Preserve the applicable source attribution and terms.
- MAGE's [dataset card](https://huggingface.co/datasets/yaful/MAGE) identifies
  Apache-2.0, while its [README badge](https://github.com/yafuly/MAGE/blob/main/README.md)
  identifies CC BY 4.0. The release does not resolve this inconsistency or
  treat a collection-level label as proof of rights to every underlying source.
- The WritingPrompts and XSum project links identify data acquisition routes;
  code licenses alone do not establish the applicable terms for their source texts.

Per-source redistribution authorization and exact source versions have not
been established for every retained text. This remains an open publication
item for the text-inclusive release. The numerical artifacts and benchmark
bytes are retained unchanged for scientific traceability. Use the original
providers' applicable terms when obtaining or reusing data.
