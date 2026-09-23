import os

import pytest
from tokenizers import Tokenizer

DEFAULT = "tokenizers/extended/fw2hq-alif-preserve-150k/tokenizer.json"
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", DEFAULT)

CASES = [
    "قال الرئيس إن الاقتصاد العالمي يواجه تحديات كبيرة في السنوات المقبلة، وأضاف أن التعاون الدولي ضروري.",
    "أعلنت الشركة عن أرباح بلغت 25% خلال الربع الأول من عام 2026.",
    "إنّ العلمَ نورٌ؛ والجهلَ ظلامٌ.",
    "الذكاء الاصطناعي يغيّر العالم.",
    "السطر الأول\nالسطر الثاني",
    "كلمة  كلمتان   ثلاث",
    " يبدأ بمسافة",
    "مزيج English و عربي في نفس الجملة.",
]


def _lep_added_tokens_start(default: int = 151669) -> int:
    import json
    cfg = os.path.join(os.path.dirname(TOKENIZER_PATH), "tokenizer_config.json")
    if os.path.exists(cfg):
        with open(cfg) as fh:
            return int(json.load(fh).get("lep_old_vocab_size", default))
    return default


@pytest.fixture(scope="module")
def tok():
    if not os.path.exists(TOKENIZER_PATH):
        pytest.skip(f"tokenizer not found: {TOKENIZER_PATH}")
    return Tokenizer.from_file(TOKENIZER_PATH)


@pytest.mark.parametrize("text", CASES, ids=range(len(CASES)))
def test_roundtrip_matches_normalized(tok, text):
    expected = tok.normalizer.normalize_str(text)
    got = tok.decode(tok.encode(text).ids)
    assert got == expected


def test_no_phantom_space_before_punctuation(tok):
    got = tok.decode(tok.encode("بلغت 25% من الناتج، وهو رقم كبير.").ids)
    assert " ," not in got
    assert " %" not in got
    assert " ." not in got


def test_words_are_not_glued(tok):
    text = "الاقتصاد العالمي يواجه تحديات"
    got = tok.decode(tok.encode(text).ids)
    assert got.split() == tok.normalizer.normalize_str(text).split()
    assert "الاقتصادالعالمي" not in got


def test_no_bare_whitespace_token_in_vocab(tok):
    added = tok.get_added_tokens_decoder()
    blank = [t.content for t in added.values() if t.content.strip() == ""
             and not t.special]
    assert blank == []


def test_no_ascii_punctuation_tokens(tok):
    start = _lep_added_tokens_start()
    leaked = [
        t.content for i, t in tok.get_added_tokens_decoder().items()
        if i >= start and not t.special and t.content.strip()
        and all(ord(c) < 128 for c in t.content)
    ]
    assert leaked == [], f"non-Arabic tokens leaked into vocab: {leaked[:20]}"
