from aratoken.normalizer import paper_normalizer


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
