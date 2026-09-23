import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--splits-dir", type=Path, default=Path("data/fineweb2hq-ar"))
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--block-size", type=int, default=1024)
    p.add_argument("--num-proc", type=int, default=32)
    p.add_argument("--train-split", default="lm-train")
    p.add_argument("--val-split", default="dev")
    p.add_argument("--max-train-tokens", type=int, default=None,
                   help="Stop after roughly this many packed tokens.")
    p.add_argument("--max-val-tokens", type=int, default=4_000_000)
    return p.parse_args()


def load_jsonl_dir(path: Path) -> Dataset:
    shards = sorted(str(p) for p in path.glob("*.jsonl"))
    if not shards:
        raise SystemExit(f"no shards under {path}")
    return load_dataset("json", data_files=shards, split="train")


def main() -> int:
    args = parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    eos_id = tok.eos_token_id
    bs = args.block_size
    print(f"tokenizer {args.tokenizer} (vocab {len(tok)}), block_size {bs}, eos {eos_id}")

    def encode(batch):
        out = tok(batch["text"], add_special_tokens=False,
                  truncation=False, padding=False)
        if eos_id is not None:
            for ids in out["input_ids"]:
                ids.append(eos_id)
            for m in out["attention_mask"]:
                m.append(1)
        return out

    def group(batch):
        cat = {k: sum(batch[k], []) for k in batch}
        total = (len(cat["input_ids"]) // bs) * bs
        res = {k: [v[i:i + bs] for i in range(0, total, bs)] for k, v in cat.items()}
        res["labels"] = [x[:] for x in res["input_ids"]]
        return res

    out = {}
    for split_name, target, budget in (
        (args.train_split, "train", args.max_train_tokens),
        (args.val_split, "validation", args.max_val_tokens),
    ):
        ds = load_jsonl_dir(args.splits_dir / split_name)
        print(f"{split_name}: {len(ds):,} docs")
        cols = ds.column_names
        ds = ds.map(encode, batched=True, batch_size=1000,
                    num_proc=args.num_proc, remove_columns=cols,
                    desc=f"tokenize:{split_name}")
        ds = ds.map(group, batched=True, batch_size=1000,
                    num_proc=args.num_proc, desc=f"pack:{split_name}")
        if budget:
            keep = min(len(ds), budget // bs)
            if keep < len(ds):
                ds = ds.select(range(keep))
        print(f"  -> {len(ds):,} blocks = {len(ds)*bs/1e6:.1f}M tokens")
        out[target] = ds

    dd = DatasetDict(out)
    args.out.mkdir(parents=True, exist_ok=True)
    dd.save_to_disk(str(args.out))

    meta = {
        "tokenizer": args.tokenizer,
        "vocab_size": len(tok),
        "block_size": bs,
        "splits_dir": str(args.splits_dir),
        "train_split": args.train_split,
        "val_split": args.val_split,
        "blocks": {k: len(v) for k, v in out.items()},
        "tokens": {k: len(v) * bs for k, v in out.items()},
        "eos_terminated": eos_id is not None,
        "python_side_normalization": False,
    }
    (args.out / "pretokenize_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
