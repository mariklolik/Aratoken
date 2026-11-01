import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from aratoken.init import build


@pytest.fixture
def init_paths(tmp_path):
    base, extended = tmp_path / "base", tmp_path / "extended"
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
        {"<unk>": 0, "<eos>": 1, "a": 2, "b": 3}, unk_token="<unk>")),
        unk_token="<unk>", eos_token="<eos>")
    tokenizer.save_pretrained(base)
    tokenizer.add_tokens(["ab", "aa", "ba", "bb"])
    tokenizer.save_pretrained(extended)
    torch.manual_seed(0)
    config = Qwen3Config(vocab_size=6, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                         tie_word_embeddings=False)
    Qwen3ForCausalLM(config).save_pretrained(base)
    return str(base), str(extended), tmp_path


@pytest.mark.parametrize("method", ["mean", "random", "calib", "wechsel", "focus", "hybrid"])
def test_initializer_preserves_original_rows_and_records_boundary(init_paths, method):
    base, tokenizer, root = init_paths
    out = root / method
    build(method, tokenizer, out, base, seed=7, topk=3, alpha=0.25)
    original = Qwen3ForCausalLM.from_pretrained(base)
    initialized = Qwen3ForCausalLM.from_pretrained(out)
    assert initialized.config.lep_added_tokens_start == 4
    assert initialized.config.lep_init_seed == 7
    assert initialized.config.lep_init_topk == 3
    assert initialized.config.lep_init_alpha == 0.25
    for getter in ("get_input_embeddings", "get_output_embeddings"):
        expected = getattr(original, getter)().weight[:4].to(torch.bfloat16).float()
        assert torch.equal(expected, getattr(initialized, getter)().weight[:4])


@pytest.mark.parametrize("alpha,method", [(0.0, "wechsel"), (1.0, "mean")])
def test_hybrid_endpoints_match_components(init_paths, alpha, method):
    base, tokenizer, root = init_paths
    build("hybrid", tokenizer, root / "hybrid", base, alpha=alpha)
    build(method, tokenizer, root / method, base)
    hybrid = Qwen3ForCausalLM.from_pretrained(root / "hybrid")
    component = Qwen3ForCausalLM.from_pretrained(root / method)
    assert torch.equal(hybrid.get_input_embeddings().weight, component.get_input_embeddings().weight)


@pytest.mark.parametrize("alpha", [-0.1, 1.1, float("nan")])
def test_invalid_alpha_fails_before_loading(tmp_path, alpha):
    with pytest.raises(ValueError, match="alpha"):
        build("hybrid", "missing", tmp_path, "missing", alpha=alpha)


def test_random_initialization_seed_is_reproducible(init_paths):
    base, tokenizer, root = init_paths
    for name, seed in (("first", 7), ("repeat", 7), ("different", 8)):
        build("random", tokenizer, root / name, base, seed=seed)
    weights = [Qwen3ForCausalLM.from_pretrained(root / name).get_input_embeddings().weight
               for name in ("first", "repeat", "different")]
    assert torch.equal(weights[0], weights[1])
    assert not torch.equal(weights[0][4:], weights[2][4:])
