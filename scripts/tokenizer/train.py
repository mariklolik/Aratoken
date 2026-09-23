import argparse
import json
import tempfile
from pathlib import Path

import sentencepiece as spm
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece
from tokenizers.trainers import BpeTrainer, WordPieceTrainer

from aratoken.normalizer import ArabicNormalizerConfig, build_qwen_like_normalizer, describe_normalizer
from aratoken.pretokenizer import ArabicPreTokenizerConfig, build_qwen_like_pretokenizer, describe_pretokenizer
from aratoken.splits import load_texts

SPECIAL_TOKENS = [
    "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|object_ref_start|>", "<|object_ref_end|>",
    "<|box_start|>", "<|box_end|>", "<|quad_start|>", "<|quad_end|>", "<|vision_start|>",
    "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>", "<tool_call>",
    "</tool_call>", "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>",
    "<|repo_name|>", "<|file_sep|>",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("algorithm", choices=["sp", "bpe", "wordpiece"])
    p.add_argument("out_dir", type=Path)
    p.add_argument("--dataset-shard-dir", type=Path, default=Path("data/fineweb2hq-ar/tok-train"))
    p.add_argument("--vocab-size", type=int, default=150_000)
    p.add_argument("--drop-diacritics", action="store_true")
    p.add_argument("--no-arabic-normalization", action="store_true",
                   help="Skip the character rules, so the plain and _norm arms differ in one step.")
    p.add_argument("--preserve-alif-forms", action="store_true")
    p.add_argument("--byte-fallback", action="store_true")
    p.add_argument("--character-coverage", type=float, default=0.9995)
    p.add_argument("--input-sentence-size", type=int, default=10_000_000)
    return p.parse_args()


def train_sentencepiece(a, texts: list[str], normalizer) -> dict:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp:
        for text in texts:
            tmp.write(normalizer.normalize_str(text).replace("\n", " ") + "\n")
    spm.SentencePieceTrainer.train(
        input=tmp.name, model_prefix=str(a.out_dir / "sentencepiece"), model_type="unigram",
        vocab_size=a.vocab_size, character_coverage=a.character_coverage,
        input_sentence_size=a.input_sentence_size, shuffle_input_sentence=True,
        pad_id=0, pad_piece="<|endoftext|>", unk_id=1, bos_id=-1, eos_id=-1,
        user_defined_symbols=SPECIAL_TOKENS[1:], hard_vocab_limit=False, byte_fallback=a.byte_fallback)
    Path(tmp.name).unlink(missing_ok=True)
    return {"algorithm": "sentencepiece", "byte_fallback": a.byte_fallback, "add_prefix_space": False,
            "pretokenizer_spec": {"type": "sentencepiece"}}


def train_hf(a, texts: list[str], normalizer_cfg) -> dict:
    pretokenizer_cfg = ArabicPreTokenizerConfig()
    if a.algorithm == "bpe":
        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        trainer = BpeTrainer(vocab_size=a.vocab_size, min_frequency=2, special_tokens=SPECIAL_TOKENS)
    else:
        tokenizer = Tokenizer(WordPiece(unk_token="<unk>"))
        trainer = WordPieceTrainer(vocab_size=a.vocab_size, min_frequency=2,
                                   special_tokens=["<unk>"] + SPECIAL_TOKENS, unk_token="<unk>")
    tokenizer.normalizer = build_qwen_like_normalizer(normalizer_cfg)
    tokenizer.pre_tokenizer = build_qwen_like_pretokenizer(pretokenizer_cfg)
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    tokenizer.save(str(a.out_dir / "tokenizer.json"))
    return {"algorithm": a.algorithm, "add_prefix_space": pretokenizer_cfg.add_prefix_space,
            "pretokenizer_spec": describe_pretokenizer(pretokenizer_cfg)}


def main() -> None:
    a = parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    normalizer_cfg = ArabicNormalizerConfig(remove_diacritics=a.drop_diacritics,
                                            preserve_alif_variants=a.preserve_alif_forms,
                                            arabic_char_replacements=not a.no_arabic_normalization)
    texts = load_texts(a.dataset_shard_dir)
    if a.algorithm == "sp":
        spec = train_sentencepiece(a, texts, build_qwen_like_normalizer(normalizer_cfg))
    else:
        spec = train_hf(a, texts, normalizer_cfg)
    config = {"tokenizer_class": "ArabicQwenTokenizer", "retain_diacritics": not a.drop_diacritics,
              "script_variant": "arabic", **spec, "clean_up_tokenization_spaces": False,
              "normalizer_spec": describe_normalizer(normalizer_cfg), "model_max_length": 131072,
              "pad_token": "<|endoftext|>", "eos_token": "<|endoftext|>",
              "additional_special_tokens": SPECIAL_TOKENS}
    (a.out_dir / "tokenizer_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2),
                                                     encoding="utf-8")
    print(f"wrote {a.out_dir}")


if __name__ == "__main__":
    main()
