import hydra
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
        list(cfg.unfreeze_transformer_layers), cfg.lora_rank, cfg.lora_alpha)
    dataset = load_from_disk(cfg.data_path)

    args = TrainingArguments(
        **OmegaConf.to_container(cfg.training, resolve=True),
        seed=cfg.seed, data_seed=cfg.seed, num_train_epochs=1, push_to_hub=False,
        gradient_checkpointing_kwargs={"use_reentrant": False}, ddp_find_unused_parameters=False)
    trainer_cls = NoDecayEmbeddingsTrainer if cfg.freeze_old_embeddings else Trainer
    trainer = trainer_cls(
        model=model, args=args, train_dataset=dataset["train"], eval_dataset=dataset["validation"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        processing_class=tokenizer)
    trainer.add_callback(EvaluateFirstStepCallback())
    trainer.train(resume_from_checkpoint=True if cfg.resume else None)
    for h in hooks:
        h.remove()


if __name__ == "__main__":
    main()
