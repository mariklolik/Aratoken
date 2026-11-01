import json

import torch
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast, TrainingArguments
from transformers import Qwen3Config, Qwen3ForCausalLM

from aratoken.training import NoDecayEmbeddingsTrainer, install_freeze_hooks, prepare_model


def test_original_rows_get_no_gradient():
    torch.manual_seed(0)
    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8, tie_word_embeddings=True)
    model = Qwen3ForCausalLM(config)
    install_freeze_hooks(model, frozen_rows=48)
    ids = torch.randint(0, 64, (2, 16))
    model(ids, labels=ids).loss.backward()
    grad = model.get_input_embeddings().weight.grad
    assert torch.count_nonzero(grad[:48]) == 0
    assert torch.count_nonzero(grad[48:]) > 0


@pytest.fixture(params=[True, False], ids=["tied", "untied"])
def tiny_model(request, tmp_path, monkeypatch):
    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                         tie_word_embeddings=request.param)
    config.lep_added_tokens_start = 48
    model = Qwen3ForCausalLM(config)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
        {f"t{i}": i for i in range(64)}, unk_token="t0")), unk_token="t0", eos_token="t1")
    tokenizer.save_pretrained(tmp_path)
    monkeypatch.setattr("aratoken.training.load_causal_lm", lambda *args: model)
    return model, str(tmp_path)


def test_cpt_embedding_training_is_explicit(tiny_model):
    model, path = tiny_model
    prepare_model(path, path, False, False, [1], device="cpu")
    assert not model.get_input_embeddings().weight.requires_grad
    model, _, _ = prepare_model(path, path, False, False, [1], device="cpu", train_embeddings=True)
    assert model.get_input_embeddings().weight.requires_grad
    assert model.get_output_embeddings().weight.requires_grad
    assert not any(p.requires_grad for p in model.model.layers[0].parameters())
    assert all(p.requires_grad for p in model.model.layers[1].parameters())
    before = model.get_input_embeddings().weight.detach().clone()
    ids = torch.tensor([[2, 3, 49, 50]])
    model(ids, labels=ids).loss.backward()
    torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1).step()
    assert not torch.equal(before[:48], model.get_input_embeddings().weight[:48])


def test_all_parameters_include_final_norm(tiny_model):
    _, path = tiny_model
    model, _, _ = prepare_model(path, path, False, False, [0, 1], device="cpu", train_embeddings=True)
    assert all(p.requires_grad for p in model.parameters())


def test_lep_old_rows_survive_adamw(tiny_model, tmp_path):
    _, path = tiny_model
    model, _, handles = prepare_model(path, path, True, True, [1], device="cpu")
    trainer = NoDecayEmbeddingsTrainer(model=model, args=TrainingArguments(
        output_dir=str(tmp_path / "out"), learning_rate=0.01, weight_decay=0.1,
        use_cpu=True, report_to="none"))
    optimizer = trainer.create_optimizer()
    tables = [model.get_input_embeddings().weight, model.get_output_embeddings().weight]
    before = [p.detach().clone() for p in tables]
    ids = torch.tensor([[2, 3, 49, 50]])
    model(ids, labels=ids).loss.backward()
    optimizer.step()
    for old, current in zip(before, tables):
        assert torch.equal(old[:48], current[:48])
        assert not torch.equal(old[48:], current[48:])
    for handle in handles:
        handle.remove()


@pytest.mark.parametrize("examples,steps", [(2, 1), (3, 2)])
def test_training_saves_final_model_and_budget(tiny_model, tmp_path, examples, steps):
    from datasets import Dataset, DatasetDict
    from omegaconf import OmegaConf
    from scripts.lep.train import main

    _, path = tiny_model
    dataset = Dataset.from_dict({"input_ids": [[2, 3, 49, 50]] * examples,
                                 "attention_mask": [[1] * 4] * examples})
    data_path = tmp_path / "data"
    DatasetDict(train=dataset, validation=dataset).save_to_disk(data_path)
    output = tmp_path / "training"
    cfg = OmegaConf.create({
        "seed": 7, "model_path": path, "tokenizer_path": path, "data_path": str(data_path),
        "freeze_old_embeddings": False, "include_lm_head_freeze": False, "train_embeddings": True,
        "unfreeze_transformer_layers": [1], "lora_rank": 0, "lora_alpha": 32, "resume": False,
        "training": {"output_dir": str(output), "max_steps": steps, "per_device_train_batch_size": 2,
                     "gradient_accumulation_steps": 1, "per_device_eval_batch_size": 2,
                     "use_cpu": True, "report_to": "none", "save_strategy": "no",
                     "load_best_model_at_end": False, "disable_tqdm": True}})
    main.__wrapped__(cfg)
    restored = Qwen3ForCausalLM.from_pretrained(output / "final_model")
    assert restored.config.vocab_size == 64
    metrics = json.loads((output / "run_metrics.json").read_text())
    assert metrics["global_step"] == steps
    assert metrics["packed_tokens"] == examples * 4
    assert metrics["checkpoint_selection"] == "final_step"
    assert metrics["train_embeddings"] is True
    assert metrics["train_runtime"] > 0
