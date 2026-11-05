import json
from pathlib import Path

import hydra
import torch
from datasets import load_from_disk
from omegaconf import DictConfig, OmegaConf
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments, set_seed

from aratoken.training import EvaluateFirstStepCallback, NoDecayEmbeddingsTrainer, prepare_model


@hydra.main(version_base=None, config_path="../../configs", config_name="lep")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)
    model, tokenizer, hooks = prepare_model(
        cfg.model_path, cfg.tokenizer_path, cfg.freeze_old_embeddings, cfg.include_lm_head_freeze,
        list(cfg.unfreeze_transformer_layers), cfg.lora_rank, cfg.lora_alpha,
        train_embeddings=cfg.train_embeddings)
    dataset = load_from_disk(cfg.data_path)

    args = TrainingArguments(
        **OmegaConf.to_container(cfg.training, resolve=True),
        seed=cfg.seed, data_seed=cfg.seed, num_train_epochs=1, push_to_hub=False,
        include_num_input_tokens_seen=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, ddp_find_unused_parameters=False)
    trainer_cls = NoDecayEmbeddingsTrainer if cfg.freeze_old_embeddings else Trainer
    trainer = trainer_cls(
        model=model, args=args, train_dataset=dataset["train"], eval_dataset=dataset["validation"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        processing_class=tokenizer)
    trainer.add_callback(EvaluateFirstStepCallback())
    use_cuda = model.device.type == "cuda"
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(model.device)
    result = trainer.train(resume_from_checkpoint=True if cfg.resume else None)
    if use_cuda:
        torch.cuda.synchronize(model.device)
    peaks = torch.tensor([
        torch.cuda.max_memory_allocated(model.device) if use_cuda else 0,
        torch.cuda.max_memory_reserved(model.device) if use_cuda else 0], device=args.device)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(peaks, op=torch.distributed.ReduceOp.MAX)
    output = Path(args.output_dir)
    trainer.save_model(str(output / "final_model"))
    trainer.save_metrics("train", result.metrics)
    if trainer.is_world_process_zero():
        block_size = len(dataset["train"][0]["input_ids"])
        global_batch = args.train_batch_size * args.gradient_accumulation_steps * args.world_size
        metrics = {**result.metrics, "global_step": trainer.state.global_step,
                   "packed_tokens": trainer.state.num_input_tokens_seen,
                   "nominal_packed_tokens": trainer.state.global_step * global_batch * block_size,
                   "block_size": block_size, "global_batch_size": global_batch,
                   "train_embeddings": cfg.train_embeddings,
                   "freeze_old_embeddings": cfg.freeze_old_embeddings,
                   "trainable_layers": list(cfg.unfreeze_transformer_layers),
                   "seed": cfg.seed, "peak_allocated_bytes": peaks[0].item(),
                   "peak_reserved_bytes": peaks[1].item(),
                   "checkpoint_selection": "best_dev" if args.load_best_model_at_end else "final_step",
                   "best_model_checkpoint": trainer.state.best_model_checkpoint}
        (output / "run_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    for h in hooks:
        h.remove()


if __name__ == "__main__":
    main()
