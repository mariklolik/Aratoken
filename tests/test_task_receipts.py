from argparse import Namespace
import json

from datasets import Dataset
import pytest

from scripts.eval import oall


def test_task_items_preserve_raw_limit_and_rejected_identity(monkeypatch):
    records = [
        {"Question": "one", "A": "a", "B": "b", "C": "c", "Answer": "ب"},
        {"Question": "two", "A": None, "B": "b", "C": "c", "Answer": "أ"},
        {"Question": "three", "A": "a", "B": "b", "C": "c", "Answer": "أ"},
    ]
    monkeypatch.setattr(oall, "load_dataset", lambda *a, **k: Dataset.from_list(records))
    items, excluded, sources = oall.task_items(
        "asas-ai/AraTrust-categorized", ["Privacy"], "train",
        lambda r: oall._letters(r, "ABC", "Answer"), 2)
    assert len(items) == 1 and items[0]["gold"] == 1
    assert items[0]["row_index"] == 0 and items[0]["config"] == "Privacy"
    assert len(excluded) == 1 and excluded[0]["row_index"] == 1
    assert sources[0]["selected_rows"] == 2


def test_task_item_identity_tracks_content_and_source_config(monkeypatch):
    records = [{"question": "one", "choices": ["a", "b"], "answer": 0}]
    monkeypatch.setattr(oall, "load_dataset", lambda *a, **k: Dataset.from_list(records))
    first, _, _ = oall.task_items("test", ["one", "two"], "test", oall._mmlu, 1)
    again, _, _ = oall.task_items("test", ["one", "two"], "test", oall._mmlu, 1)
    assert first == again
    assert first[0]["item_sha256"] != first[1]["item_sha256"]
    records[0]["question"] = "changed"
    changed, _, _ = oall.task_items("test", ["one"], "test", oall._mmlu, 1)
    assert first[0]["item_sha256"] != changed[0]["item_sha256"]


def test_evaluation_rejects_stale_cache(monkeypatch, tmp_path):
    class Model:
        def to(self, device):
            return self

        def eval(self):
            return self

    args = Namespace(model="new-model", tokenizer="tokenizer", tag="x", out=tmp_path,
                     device="cpu", lang="en", protocol="mcq", limit=1)
    monkeypatch.setattr(oall, "parse_args", lambda: args)
    monkeypatch.setattr(oall, "resolve_artifact", lambda path: tmp_path)
    monkeypatch.setattr(oall, "artifact_hash", lambda path, kind: "0" * 64)
    monkeypatch.setattr(oall.AutoTokenizer, "from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr(oall.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: Model())
    monkeypatch.setattr(oall, "EN_TASKS", [("MMLU", "test", ["all"], "test", oall._mmlu)])
    monkeypatch.setattr(oall, "load_dataset", lambda *a, **k: Dataset.from_list([
        {"question": "one", "choices": ["a", "b"], "answer": 0}]))
    out = tmp_path / "mcq" / "en"
    out.mkdir(parents=True)
    (out / "x_MMLU.json").write_text(json.dumps({
        "model": "old-model", "accuracy": 1.0, "n": 1, "y_true": [0], "y_pred": [0]}))
    with pytest.raises(ValueError, match="cache"):
        oall.main()


def test_model_digest_detects_replaced_weights_at_same_path(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"first")
    before = oall.artifact_hash(tmp_path, "model")
    weights.write_bytes(b"second")
    assert oall.artifact_hash(tmp_path, "model") != before


def test_tokenizer_digest_detects_replaced_tokenizer_at_same_path(tmp_path):
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text('{"version":1}')
    before = oall.artifact_hash(tmp_path, "tokenizer")
    tokenizer.write_text('{"version":2}')
    assert oall.artifact_hash(tmp_path, "tokenizer") != before
