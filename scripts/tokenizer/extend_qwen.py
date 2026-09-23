import argparse

from aratoken.normalizer import ArabicNormalizerConfig
from aratoken.pretokenizer import ArabicPreTokenizerConfig
from aratoken.vocab import extend_tokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sentencepiece", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--base-model", default="Qwen/Qwen3-0.6B-Base")
    p.add_argument("--unify-alif", action="store_true")
    p.add_argument("--keep-diacritics", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    extend_tokenizer(
        a.base_model, a.sentencepiece, a.out,
        ArabicNormalizerConfig(unicode_normalization="NFKC", remove_diacritics=not a.keep_diacritics,
                               lowercase_latin=True, preserve_alif_variants=not a.unify_alif),
        ArabicPreTokenizerConfig(with_prefix_space=False, add_prefix_space=False,
                                 fuse_numeric_sequences=True))
