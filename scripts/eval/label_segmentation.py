import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from scripts.eval.classification import DATASETS, LETTERS


def label_costs(tok) -> dict:
    arm = {name: {lbl: {"tokens": len(tok(" " + lbl, add_special_tokens=False).input_ids),
                        "bytes": len((" " + lbl).encode("utf-8"))}
                  for lbl in cfg["labels"].values()}
           for name, cfg in DATASETS.items()}
    arm["_letters"] = {c: len(tok(" " + c, add_special_tokens=False).input_ids) for c in LETTERS}
    return arm


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arms", nargs="+", required=True, metavar="NAME=TOKENIZER")
    p.add_argument("--out", type=Path, default=Path("results/label_segmentation.json"))
    a = p.parse_args()
    out = {name: label_costs(AutoTokenizer.from_pretrained(path))
           for name, path in (spec.split("=", 1) for spec in a.arms)}
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, arm in out.items():
        costs = [v["tokens"] for ds, d in arm.items() if ds != "_letters" for v in d.values()]
        print(f"{name:10} label words {min(costs)}-{max(costs)} tokens, "
              f"letters {sorted(set(arm['_letters'].values()))} tokens")


if __name__ == "__main__":
    main()
