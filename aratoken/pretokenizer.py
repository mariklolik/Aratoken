from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from tokenizers import pre_tokenizers

ARABIC_AWARE_SPLIT = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?\p{Script=Arabic}+"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}+"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


@dataclass
class ArabicPreTokenizerConfig:
    with_prefix_space: bool = False
    add_prefix_space: bool = True
    fuse_numeric_sequences: bool = True


def build_qwen_like_pretokenizer(config: ArabicPreTokenizerConfig):
    steps = [
        pre_tokenizers.Split(pattern=ARABIC_AWARE_SPLIT, behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=config.add_prefix_space, use_regex=True),
    ]

    if config.fuse_numeric_sequences:
        steps.insert(1, pre_tokenizers.Digits())

    if config.with_prefix_space:
        steps.insert(0, pre_tokenizers.Sequence([pre_tokenizers.Whitespace()]))

    return pre_tokenizers.Sequence(steps)


def describe_pretokenizer(config: ArabicPreTokenizerConfig) -> Dict[str, Any]:
    return {
        "with_prefix_space": config.with_prefix_space,
        "add_prefix_space": config.add_prefix_space,
        "fuse_numeric_sequences": config.fuse_numeric_sequences,
    }
