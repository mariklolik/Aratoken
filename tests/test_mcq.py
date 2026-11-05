import pytest
from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from scripts.eval.classification import DATASETS, LETTERS, mcq_prompt
from scripts.eval.label_segmentation import label_costs
from scripts.eval.oall import EN_LETTERS, _letters, _mbzuai, as_mcq


@pytest.mark.parametrize("rotation", range(7))
def test_sanad_all_candidates_and_demonstrations_rotate(rotation):
    cfg = DATASETS["sanad"]
    labels = list(cfg["labels"].values())
    demos = [(f"demo{i}", label) for i, label in enumerate(labels)]
    prompt, letters, order = mcq_prompt(cfg, "target", demos, rotation)
    assert letters == ["أ", "ب", "ج", "د", "ه", "و", "ز"]
    assert order == labels[rotation:] + labels[:rotation]
    assert len(set(order)) == 7
    for i, label in enumerate(labels):
        letter = letters[order.index(label)]
        assert f"{letter}. {label}\n" in prompt
        assert f'مقالة: "demo{i}"\nالتقييم: {letter}\n' in prompt


def test_four_choice_labels_keep_existing_order():
    prompt, letters, order = mcq_prompt(DATASETS["ASTD"], "target", [], 2)
    assert letters == ["أ", "ب", "ج", "د"]
    assert order == ["إيجابي", "سلبي", "حيادي", "موضوعي"]
    assert "أ. إيجابي\nب. سلبي\nج. حيادي\nد. موضوعي" in prompt
    assert prompt.endswith('النص: "target"\nالتسمية:')


@pytest.mark.parametrize("count", [0, 8])
def test_classification_rejects_unsupported_choice_count(count):
    cfg = {**DATASETS["ASTD"], "labels": {i: f"label{i}" for i in range(count)}}
    with pytest.raises(ValueError, match="choice"):
        mcq_prompt(cfg, "target", [], 0)


@pytest.mark.parametrize("letters,count", [(LETTERS, 0), (LETTERS, 8), (EN_LETTERS, 6)])
def test_oall_rejects_unsupported_choice_count(letters, count):
    with pytest.raises(ValueError, match="choice"):
        as_mcq("question\nanswer:", [f"choice{i}" for i in range(count)], letters)


def test_oall_shared_arabic_alphabet_preserves_tail_and_gold_mapping():
    prompt, letters, order = as_mcq("السؤال: question\nالإجابة:", list("1234567"), LETTERS, 6)
    assert letters == ["أ", "ب", "ج", "د", "ه", "و", "ز"]
    assert order == [6, 0, 1, 2, 3, 4, 5]
    assert "أ. 7\nب. 1\nج. 2\nد. 3\nه. 4\nو. 5\nز. 6" in prompt
    assert prompt.endswith("الإجابة:")


def test_oall_english_tail_is_unchanged():
    prompt, letters, order = as_mcq("Question: q\nAnswer:", ["one", "two"], EN_LETTERS, 1)
    assert prompt == "Question: q\nA. two\nB. one\nAnswer:"
    assert letters == ["A", "B"]
    assert order == [1, 0]


@pytest.mark.parametrize("answer,expected", [("A", None), ("B", 0), ("C", 1), ("", None)])
def test_letter_adapter_maps_gold_through_present_options(answer, expected):
    item = _letters({"Question": "q", "A": None, "B": "second", "C": "third", "Answer": answer},
                    "ABC", "Answer")
    if expected is None:
        assert item is None
    else:
        assert item == ("السؤال: q\nالإجابة:", ["second", "third"], expected)


@pytest.mark.parametrize("answer,expected", [("A", None), ("B", 0), ("C", 1), ("", None)])
def test_mbzuai_adapter_maps_gold_through_present_options(answer, expected):
    item = _mbzuai({"Question": "q", "Context": "context", "Option 1": "None", "Option 2": "second",
                    "Option 3": "third", "Option 4": "", "Option 5": None, "Answer Key": answer})
    if expected is None:
        assert item is None
    else:
        assert item == ("context\nالسؤال: q\nالإجابة:", ["second", "third"], expected)


def test_label_costs_cover_all_seven_equal_byte_letters():
    backend = Tokenizer(WordLevel({"<unk>": 0, "أ": 1, "ب": 2, "ج": 3, "د": 4,
                                   "ه": 5, "و": 6, "ز": 7}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>")
    assert label_costs(tok)["_letters"] == {"أ": 1, "ب": 1, "ج": 1, "د": 1, "ه": 1, "و": 1, "ز": 1}
    assert {len((" " + letter).encode("utf-8")) for letter in LETTERS} == {3}
