import hashlib
import math

import pytest
import torch
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from aratoken.metrics import per_doc_bpc


@pytest.fixture
def model():
    config = Qwen3Config(vocab_size=8, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2,
                         num_key_value_heads=1, head_dim=8, max_position_embeddings=32)
    result = Qwen3ForCausalLM(config).eval()
    for parameter in result.parameters():
        parameter.data.zero_()
    return result


def tokenizer(character_level=True, normalize=None):
    backend = Tokenizer(models.WordLevel({"[UNK]": 0, "[BOS]": 1, "a": 2,
                                          "b": 3, "ab": 4, "[EOS]": 5}, unk_token="[UNK]"))
    backend.pre_tokenizer = (pre_tokenizers.Split("", behavior="isolated") if character_level
                             else pre_tokenizers.Whitespace())
    backend.normalizer = normalize
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
                                   bos_token="[BOS]", eos_token="[EOS]")


def test_common_text_scores_every_target_and_uses_identical_character_denominators(model):
    fine, fine_totals = per_doc_bpc(model, tokenizer(), ["ABaa", "ab"], "cpu", 32,
                                   common_text_chars=2)
    coarse, coarse_totals = per_doc_bpc(model, tokenizer(False), ["ABaa", "ab"], "cpu", 32,
                                       common_text_chars=2)
    assert fine.tolist() == pytest.approx([3.0, 3.0])
    assert coarse.tolist() == pytest.approx([1.5, 1.5])
    assert fine_totals["total_chars"] == coarse_totals["total_chars"] == 4
    assert fine_totals["total_tokens"] == 4
    assert coarse_totals["total_tokens"] == 2
    assert [d["text_sha256"] for d in fine_totals["documents"]] == [
        hashlib.sha256(b"ab").hexdigest()] * 2
    assert fine_totals["documents"][0]["source_sha256"] == hashlib.sha256(b"ABaa").hexdigest()
    assert fine_totals["documents"][0]["bpc"] == pytest.approx(3.0)
    assert fine_totals["protocol"]["prefix_token_id"] == 1


def test_common_text_rejects_context_overflow_instead_of_truncating(model):
    with pytest.raises(ValueError, match="context"):
        per_doc_bpc(model, tokenizer(), ["aaaa"], "cpu", 4, common_text_chars=4)


def test_common_text_rejects_tokenizer_changes_to_canonical_text(model):
    tok = tokenizer(normalize=normalizers.Replace("a", "b"))
    with pytest.raises(ValueError, match="normaliz"):
        per_doc_bpc(model, tok, ["ab"], "cpu", 32, common_text_chars=2)


def test_common_text_uses_eos_when_bos_is_absent(model):
    tok = tokenizer()
    tok.bos_token = None
    values, totals = per_doc_bpc(model, tok, ["a"], "cpu", 32, common_text_chars=1)
    assert values[0] == pytest.approx(3.0)
    assert totals["protocol"]["prefix_token_id"] == 5


def test_common_text_rejects_missing_prefix(model):
    tok = tokenizer()
    tok.bos_token = tok.eos_token = None
    with pytest.raises(ValueError, match="BOS|EOS"):
        per_doc_bpc(model, tok, ["ab"], "cpu", 32, common_text_chars=2)


def test_common_text_rejects_empty_canonical_document(model):
    with pytest.raises(ValueError, match="empty"):
        per_doc_bpc(model, tokenizer(), ["ـ"], "cpu", 32, common_text_chars=2)


def test_legacy_mode_preserves_unscored_first_token(model):
    values, totals = per_doc_bpc(model, tokenizer(False), ["a b"], "cpu", 32)
    assert values[0] == pytest.approx(math.log(8) / (3 * math.log(2)))
    assert "documents" not in totals


def test_common_text_canonicalizes_turkish_dot_before_tokenizer_validation(model):
    from aratoken.normalizer import paper_normalizer

    tok = tokenizer(normalize=paper_normalizer())
    _, totals = per_doc_bpc(model, tok, ["TÜİK"], "cpu", 32, common_text_chars=4)
    assert totals["total_chars"] == 4
    assert totals["documents"][0]["text_sha256"] == hashlib.sha256("tüik".encode()).hexdigest()
    assert "fixed point" in totals["protocol"]["normalizer"]
