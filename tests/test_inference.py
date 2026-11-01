import importlib
import math

import pytest


def test_prompts_canonicalize_turkish_dot_before_slicing():
    from scripts.eval.inference import canonical_prompts
    from aratoken.normalizer import paper_normalizer

    prompts = canonical_prompts(["TÜİK"], 1, 4)
    assert prompts == ["tüik"]
    assert paper_normalizer().normalize_str(prompts[0]) == prompts[0]


def test_prompts_share_canonical_text_and_exact_character_budget():
    benchmark = importlib.import_module("scripts.eval.inference")
    prompts = benchmark.canonical_prompts(["ـــ", "أ" * 5 + "ب" * 5, "ج" * 10], 2, 8)
    assert prompts == ["أأأأأببب", "جججججججج"]


@pytest.mark.parametrize("texts,count,chars", [([], 1, 8), (["ـــ"], 1, 1),
                                              (["abc"], 0, 1), (["abc"], 1, 0)])
def test_prompts_reject_insufficient_or_invalid_character_budgets(texts, count, chars):
    benchmark = importlib.import_module("scripts.eval.inference")
    with pytest.raises(ValueError):
        benchmark.canonical_prompts(texts, count, chars)


def batch(seconds, tokens, chars):
    return {"generation_seconds": seconds, "end_to_end_seconds": seconds + 1,
            "prefill_seconds": 0.5, "decode_seconds": seconds - 0.5,
            "tokenization_seconds": 0.25, "detokenization_seconds": 0.25,
            "generated_tokens": tokens, "decode_tokens": tokens - 1,
            "generated_chars": chars, "input_tokens": 8, "padded_input_tokens": 8,
            "batch_size": 1, "peak_allocated_bytes": 100, "peak_reserved_bytes": 200}


def test_summary_uses_total_work_over_total_time_not_mean_batch_rates():
    benchmark = importlib.import_module("scripts.eval.inference")
    result = benchmark.summarize_batches([batch(1, 10, 20), batch(3, 10, 40)])
    assert result["generated_tokens"] == 20
    assert result["generated_chars"] == 60
    assert result["generation_tokens_per_second"] == pytest.approx(5)
    assert result["generation_chars_per_second"] == pytest.approx(15)
    assert result["end_to_end_tokens_per_second"] == pytest.approx(20 / 6)
    assert result["decode_tokens_per_second"] == pytest.approx(6)
    assert result["peak_reserved_bytes"] == 200


@pytest.mark.parametrize("records", [[], [batch(0, 10, 20)], [batch(math.nan, 10, 20)],
                                     [batch(1, -1, 20)]])
def test_summary_rejects_invalid_measurements(records):
    benchmark = importlib.import_module("scripts.eval.inference")
    with pytest.raises(ValueError):
        benchmark.summarize_batches(records)


@pytest.mark.parametrize("flag,value", [("--limit", "0"), ("--prompt-chars", "0"),
                                       ("--new-tokens", "0"), ("--warmup", "0"),
                                       ("--batch-sizes", "0"), ("--device", "cpu")])
def test_cli_rejects_invalid_benchmark_settings(flag, value):
    benchmark = importlib.import_module("scripts.eval.inference")
    with pytest.raises(SystemExit):
        benchmark.parse_args(["--model", "test=model", "--out", "out.json", flag, value])


def test_cli_rejects_duplicate_model_labels():
    benchmark = importlib.import_module("scripts.eval.inference")
    with pytest.raises(SystemExit):
        benchmark.parse_args(["--model", "same=a", "--model", "same=b", "--out", "out.json"])


@pytest.fixture
def tiny_cuda_model_and_tokenizer():
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    torch.manual_seed(42)
    backend = Tokenizer(models.WordLevel({"[UNK]": 0, "[BOS]": 1, "a": 2,
                                         "b": 3, "c": 4, "[EOS]": 5}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
                                       bos_token="[BOS]", eos_token="[EOS]", pad_token="[EOS]",
                                       padding_side="left")
    config = Qwen3Config(vocab_size=6, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                         head_dim=8, max_position_embeddings=32, bos_token_id=1,
                         eos_token_id=5, pad_token_id=5)
    return Qwen3ForCausalLM(config).cuda().eval(), tokenizer


def test_fixed_greedy_cache_loop_matches_generate_with_left_padding(tiny_cuda_model_and_tokenizer):
    import torch

    benchmark = importlib.import_module("scripts.eval.inference")
    model, tokenizer = tiny_cuda_model_and_tokenizer
    prompts = ["aaaa", "ab"]
    result = benchmark.run_batch(model, tokenizer, prompts, "cuda:0", 4)
    inputs = tokenizer(prompts, padding=True, add_special_tokens=False, return_token_type_ids=False,
                       return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        expected = model.generate(**inputs, do_sample=False, max_new_tokens=4,
                                  min_new_tokens=4, eos_token_id=None, pad_token_id=5)
    assert result["generated_token_ids"] == expected[:, inputs.input_ids.shape[1]:].tolist()
    assert result["generated_tokens"] == 8
    assert result["decode_tokens"] == 6
    assert result["input_tokens"] == 6
    assert result["padded_input_tokens"] == 8
    assert result["end_to_end_seconds"] >= result["generation_seconds"] > 0
    assert result["peak_reserved_bytes"] >= result["peak_allocated_bytes"] > 0


def test_generation_rejects_context_overflow(tiny_cuda_model_and_tokenizer):
    benchmark = importlib.import_module("scripts.eval.inference")
    model, tokenizer = tiny_cuda_model_and_tokenizer
    with pytest.raises(ValueError, match="context"):
        benchmark.run_batch(model, tokenizer, ["a" * 30], "cuda:0", 4)
