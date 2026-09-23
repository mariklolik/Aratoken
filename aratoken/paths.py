from __future__ import annotations

SP = "tokenizers"
LEP = "tokenizers/extended"
SPLITS = "data/fineweb2hq-ar"
PRETOKENIZED = "data/pretokenized"
MODELS = "models"

ARATOKEN = {
    "AraToken-150k-alif-preserve": (f"{SP}/fw2hq_sp_alif_preserve/sentencepiece.model", True),
    "AraToken-150k-alif-preserve-bytes": (f"{SP}/fw2hq_sp_alif_preserve_bf/sentencepiece.model", True),
    "AraToken-150k-alif-unified": (f"{SP}/fw2hq_sp_alif_unified/sentencepiece.model", False),
    "AraToken-99pct-alif-unified": (f"{SP}/fw2hq_sp_alif_unified_0_99/sentencepiece.model", False),
    "AraToken-95pct-alif-unified": (f"{SP}/fw2hq_sp_alif_unified_0_95/sentencepiece.model", False),
    "AraToken-99pct-alif-preserve": (f"{SP}/fw2hq_sp_alif_preserve_0_99/sentencepiece.model", True),
    "AraToken-95pct-alif-preserve": (f"{SP}/fw2hq_sp_alif_preserve_0_95/sentencepiece.model", True),
}

EXTENDED = {
    "LEP-Qwen3+AraToken-alif-preserve": f"{LEP}/fw2hq-alif-preserve-150k",
    "LEP-Qwen3+AraToken-alif-unified": f"{LEP}/fw2hq-alif-unified-150k",
}

BASELINES = {
    "Qwen2.5": "Qwen/Qwen2.5-0.5B",
    "Qwen3": "Qwen/Qwen3-0.6B-Base",
    "DeepSeek-V3": "deepseek-ai/DeepSeek-V3",
    "AraBERT": "aubmindlab/bert-base-arabertv2",
}

PRUNING_LADDER = [
    ("full", ARATOKEN["AraToken-150k-alif-preserve"][0]),
    ("99%", ARATOKEN["AraToken-99pct-alif-preserve"][0]),
    ("95%", ARATOKEN["AraToken-95pct-alif-preserve"][0]),
]
