from __future__ import annotations

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback


def make_row_zero_hook(frozen_rows: int):
    def hook(grad: torch.Tensor):
        grad[:frozen_rows] = 0.0
        return grad
    return hook


def install_freeze_hooks(model, frozen_rows: int, include_lm_head: bool = True) -> list:
    hook = make_row_zero_hook(frozen_rows)
    input_emb = model.get_input_embeddings()
    handles = [input_emb.weight.register_hook(hook)]
    output_emb = model.get_output_embeddings()
    if include_lm_head and output_emb is not None and output_emb.weight is not input_emb.weight:
        handles.append(output_emb.weight.register_hook(hook))
    return handles


class NoDecayEmbeddingsTrainer(Trainer):
    def create_optimizer(self):
        if self.optimizer is None:
            decay = set(self.get_decay_parameter_names(self.model))
            trainable = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
            groups = [
                {"params": [p for n, p in trainable if n in decay], "weight_decay": self.args.weight_decay},
                {"params": [p for n, p in trainable if n not in decay], "weight_decay": 0.0},
            ]
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(groups, **optimizer_kwargs)
        return self.optimizer

    def get_decay_parameter_names(self, model):
        embeddings = {id(module.weight) for module in
                      (model.get_input_embeddings(), model.get_output_embeddings()) if module is not None}
        return [name for name, p in model.named_parameters()
                if p.requires_grad and id(p) not in embeddings and "embed" not in name.lower() and "bias" not in name
                and "norm" not in name.lower() and p.ndim >= 2]


class EvaluateFirstStepCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 1:
            control.should_evaluate = True


def load_causal_lm(path: str, device: str, dtype=torch.bfloat16):
    try:
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=dtype, attn_implementation="flash_attention_2", trust_remote_code=True)
    except ImportError:
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=dtype, attn_implementation="sdpa", trust_remote_code=True)
    return model.to(device)


def unfreeze_layers(model, layer_ids: list[int]) -> None:
    layers = model.model.layers
    for idx in sorted(set(layer_ids)):
        if not 0 <= idx < len(layers):
            raise ValueError(f"layer {idx} out of range ({len(layers)} layers)")
        for p in layers[idx].parameters():
            p.requires_grad = True


def freeze_all(model) -> None:
    for p in model.parameters():
        p.requires_grad = False


def new_rows_start(model, tokenizer) -> int:
    start = getattr(model.config, "lep_added_tokens_start", None)
    if start is not None:
        return start
    added = tokenizer.get_added_vocab()
    if not added:
        raise ValueError("model config has no lep_added_tokens_start and the tokenizer adds no tokens")
    return min(added.values())


def prepare_model(model_path: str, tokenizer_path: str, freeze_old_embeddings: bool,
                  include_lm_head_freeze: bool, trainable_layers: list[int],
                  lora_rank: int = 0, lora_alpha: int = 32, device: str | None = None,
                  train_embeddings: bool = False):
    device = device or (f"cuda:{os.environ.get('LOCAL_RANK', '0')}" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    model = load_causal_lm(model_path, device)
    rows = model.get_input_embeddings().weight.shape[0]
    if (freeze_old_embeddings and len(tokenizer) != rows) or len(tokenizer) > rows:
        raise ValueError(f"tokenizer has {len(tokenizer)} tokens, model has {rows} embedding rows")

    handles = []
    if freeze_old_embeddings or trainable_layers:
        freeze_all(model)
    if freeze_old_embeddings or train_embeddings:
        model.get_input_embeddings().weight.requires_grad = True
        output = model.get_output_embeddings()
        if output is not None and (include_lm_head_freeze or train_embeddings):
            output.weight.requires_grad = True
    if freeze_old_embeddings:
        handles = install_freeze_hooks(model, new_rows_start(model, tokenizer), include_lm_head_freeze)
    if freeze_old_embeddings or trainable_layers:
        unfreeze_layers(model, trainable_layers)
        model.enable_input_require_grads()
    if train_embeddings and not freeze_old_embeddings and set(trainable_layers) == set(range(len(model.model.layers))):
        model.requires_grad_(True)

    if lora_rank:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            r=lora_rank, lora_alpha=lora_alpha, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        if freeze_old_embeddings:
            model.get_input_embeddings().weight.requires_grad = True
            model.enable_input_require_grads()

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"parameters: {total:,} total, {trainable:,} trainable ({100 * trainable / total:.2f}%)")
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, handles
