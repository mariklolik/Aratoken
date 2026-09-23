# Data

All experiments use the Arabic part (`arb_Arab`) of
[FineWeb2-HQ](https://huggingface.co/datasets/epfml/FineWeb2-HQ), which is released under ODC-By 1.0.
The corpus text is not copied into this repository. It is downloaded from the Hugging Face Hub and split
by `scripts/data/build_splits.py`.

## Split rule

A document goes to a split according to `blake2b(id, digest_size=8) mod 1000`
(`aratoken/splits.py`). The mapping depends only on the document id, so the four splits are disjoint
by construction.

| split | buckets | role | documents | characters |
|---|---|---|---:|---:|
| `lm-train` | 0–599 | LEP and CPT adaptation | 662,685 | 2.00 B |
| `tok-train` | 600–899 | tokenizer training and pruning frequencies | 159,826 | 480 M |
| `dev` | 900–949 | training-loss monitoring | 2,522 | 8 M |
| `test` | 950–999 | every reported measurement | 6,602 | 20 M |

The builder reads every third parquet shard and stops each split at its character budget. Worker
threads decide which documents reach a budget first, so the exact membership of `lm-train` and
`tok-train` can differ from one rebuild to the next. The evaluation splits are released exactly:

| file | content |
|---|---|
| `splits/manifest.json` | source shards, filters, budgets and split sizes of the paper's build |
| `splits/test_ids.txt` | the 6,602 FineWeb2-HQ ids of the `test` split, in evaluation order |
| `splits/dev_ids.txt` | the 2,522 ids of the `dev` split |

```bash
python -m scripts.data.build_splits --out data/fineweb2hq-ar
python -m scripts.data.build_splits --out data/fineweb2hq-ar-eval \
    --ids data/splits/test_ids.txt data/splits/dev_ids.txt
```

The first command builds all four splits (about 3.3 GB of JSONL). The second command rebuilds only
`test` and `dev`, with the same documents in the same order as in the paper. The evaluation scripts
read the first N documents of `test`, so the order matters.

## Other data

The evaluation scripts load every other dataset from the Hugging Face Hub. Dialectal text comes from
ArSarcasm (`iabufarha/ar_sarcasm`), task accuracy from the OALL v2 native-Arabic suite (ArabicMMLU,
MadinahQA, Arabic EXAMS, AraTrust, AlGhafa), classification from ASTD, ArSarcasm, SANAD and
XNLI-ar, and English retention from WikiText-103, MMLU and ARC-Easy.
