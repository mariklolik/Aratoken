import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from aratoken.training import install_freeze_hooks


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
