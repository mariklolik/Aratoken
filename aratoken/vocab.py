from __future__ import annotations

import re

import sentencepiece as spm
from tokenizers import decoders
from transformers import AddedToken, AutoTokenizer

from aratoken.normalizer import ArabicNormalizerConfig, build_qwen_like_normalizer
from aratoken.pretokenizer import ArabicPreTokenizerConfig, build_qwen_like_pretokenizer

ARABIC_SCRIPT = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")
ARABIC_NON_LETTER = frozenset("،؍؛؟ـ٪٫٬٭۔٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹")


def is_bad_token(token: str) -> bool:
    if not token:
        return True
    if token[0] in ARABIC_NON_LETTER:
        return True
    return ARABIC_SCRIPT.match(token[0]) is None


def arabic_pieces(sentencepiece_path: str) -> tuple[set[str], set[str]]:
    sp = spm.SentencePieceProcessor()
    sp.load(sentencepiece_path)
    word_initial, word_internal = set(), set()
    for i in range(sp.get_piece_size()):
        piece = sp.id_to_piece(i)
        body = piece[1:] if piece.startswith("▁") else piece
        if not body or body.isspace() or is_bad_token(body):
            continue
        if piece.startswith("▁"):
            word_initial.add(f" {body}")
        else:
            word_internal.add(body)
    return word_initial, word_internal


def extend_tokenizer(base_model: str, sentencepiece_path: str, out_dir: str,
                     normalizer_config: ArabicNormalizerConfig,
                     pretokenizer_config: ArabicPreTokenizerConfig):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    old_size = len(tokenizer)
    word_initial, word_internal = arabic_pieces(sentencepiece_path)
    new_word_initial = word_initial - set(tokenizer.get_vocab())
    tokenizer.add_tokens([AddedToken(p, lstrip=False) for p in sorted(new_word_initial)])
    tokenizer.add_tokens([AddedToken(p) for p in sorted(word_internal)])
    print(f"added {len(tokenizer) - old_size} tokens to {base_model} ({old_size} -> {len(tokenizer)})")

    tokenizer.init_kwargs["lep_old_vocab_size"] = old_size
    tokenizer.init_kwargs["lep_new_vocab_size"] = len(tokenizer)
    tokenizer.init_kwargs["lep_sentencepiece_path"] = sentencepiece_path
    backend = tokenizer.backend_tokenizer
    backend.normalizer = build_qwen_like_normalizer(normalizer_config)
    backend.pre_tokenizer = build_qwen_like_pretokenizer(pretokenizer_config)
    backend.decoder = decoders.ByteLevel(add_prefix_space=pretokenizer_config.add_prefix_space,
                                         use_regex=True)
    tokenizer.save_pretrained(out_dir)
    return tokenizer
