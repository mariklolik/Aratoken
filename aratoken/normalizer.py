from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from tokenizers import normalizers

COMMON_CHAR_REPLACEMENTS = {
    '﴿': ')',
    '﴾': '(',
    '۔': '.',
    '٪': '%',
    '٫': '.',
    '٬': ',',
    '۹': '9',
    '۸': '8',
    '۷': '7',
    '۶': '6',
    '۵': '5',
    '۴': '4',
    '۳': '3',
    '۲': '2',
    '۱': '1',
    '۰': '0',
    '٩': '9',
    '٨': '8',
    '٧': '7',
    '٦': '6',
    '٥': '5',
    '٤': '4',
    '٣': '3',
    '٢': '2',
    '١': '1',
    '٠': '0',
    '؟': '?',
    '؛': ';',
    '؍': '/',
    '،': ',',
    'ـ': '',
}

ALIF_VARIANT_REPLACEMENTS = {
    'أ': 'ا',
    'إ': 'ا',
    'آ': 'ا',
    'ٱ': 'ا',
}


def build_char_replacements(preserve_alif_variants: bool) -> Dict[str, str]:
    replacements = dict(COMMON_CHAR_REPLACEMENTS)
    if not preserve_alif_variants:
        replacements.update(ALIF_VARIANT_REPLACEMENTS)
    return replacements


@dataclass
class ArabicNormalizerConfig:
    unicode_normalization: str = "NFKC"
    remove_diacritics: bool = False
    lowercase_latin: bool = True
    preserve_alif_variants: bool = False
    arabic_char_replacements: bool = True


def build_qwen_like_normalizer(config: ArabicNormalizerConfig):
    steps = [normalizers.NFKC()]

    if config.unicode_normalization.upper() != "NFKC":
        norm_cls = getattr(normalizers, config.unicode_normalization.upper(), None)
        if norm_cls is None:
            raise ValueError(f"Unsupported normalization: {config.unicode_normalization}")
        steps.append(norm_cls())

    if config.arabic_char_replacements:
        replacements = build_char_replacements(config.preserve_alif_variants)
        for old_char, new_char in replacements.items():
            steps.append(normalizers.Replace(old_char, new_char))

    if config.remove_diacritics:
        steps.append(normalizers.StripAccents())

    if config.lowercase_latin:
        steps.append(normalizers.Lowercase())

    return normalizers.Sequence(steps)


def paper_normalizer(preserve_alif: bool = True):
    return build_qwen_like_normalizer(ArabicNormalizerConfig(
        unicode_normalization="NFKC", remove_diacritics=True,
        lowercase_latin=True, preserve_alif_variants=preserve_alif))


def describe_normalizer(config: ArabicNormalizerConfig) -> Dict[str, Any]:
    return {
        "unicode_normalization": config.unicode_normalization,
        "remove_diacritics": config.remove_diacritics,
        "lowercase_latin": config.lowercase_latin,
        "preserve_alif_variants": config.preserve_alif_variants,
        "arabic_char_replacements":
            build_char_replacements(config.preserve_alif_variants)
            if config.arabic_char_replacements else {},
    }


def config_from_spec(spec: Dict[str, Any]) -> ArabicNormalizerConfig:
    return ArabicNormalizerConfig(
        unicode_normalization=spec.get("unicode_normalization", "NFKC"),
        remove_diacritics=spec.get("remove_diacritics", False),
        lowercase_latin=spec.get("lowercase_latin", True),
        preserve_alif_variants=spec.get("preserve_alif_variants", False),
        arabic_char_replacements=bool(spec.get("arabic_char_replacements", True)),
    )
