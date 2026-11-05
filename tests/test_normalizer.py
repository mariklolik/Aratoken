from aratoken.normalizer import paper_normalizer

import pytest


def test_character_rules():
    norm = paper_normalizer(preserve_alif=True)
    assert norm.normalize_str("كـــتاب") == "كتاب"
    assert norm.normalize_str("٢٠٢٦") == "2026"
    assert norm.normalize_str("نعم، لا؛ لماذا؟") == "نعم, لا; لماذا?"


def test_diacritics_are_removed():
    assert paper_normalizer().normalize_str("إنّ العلمَ نورٌ") == "إن العلم نور"


def test_alif_variants_are_kept_by_default_and_unified_on_request():
    text = "أحمد إلى آمن ٱلله"
    assert paper_normalizer(preserve_alif=True).normalize_str(text) == text
    assert paper_normalizer(preserve_alif=False).normalize_str(text) == "احمد الى امن الله"


def test_existing_normalizer_introduces_turkish_combining_dot_before_second_pass():
    normalizer = paper_normalizer()
    first = normalizer.normalize_str("TÜİK")
    assert first == "tüi\u0307k"
    assert normalizer.normalize_str(first) == "tüik"


def test_canonical_text_reaches_fixed_point_before_character_slicing():
    from aratoken.normalizer import canonical_text

    assert canonical_text("TÜİK", 4) == "tüik"
    assert canonical_text("TÜİK", 3) == "tüi"
    text = canonical_text("TÜİK إنّ العلمَ ٢٠٢٦", 512)
    assert paper_normalizer().normalize_str(text) == text
    assert canonical_text(text, 512) == text


@pytest.mark.parametrize("limit", [0, -1])
def test_canonical_text_rejects_nonpositive_character_budget(limit):
    from aratoken.normalizer import canonical_text

    with pytest.raises(ValueError, match="positive"):
        canonical_text("abc", limit)


def test_canonical_text_rejects_nonconverging_normalizer(monkeypatch):
    import aratoken.normalizer as module
    from tokenizers import normalizers

    monkeypatch.setattr(module, "paper_normalizer", lambda **kwargs: normalizers.Replace("a", "aa"))
    with pytest.raises(ValueError, match="converg"):
        module.canonical_text("a", 10)
