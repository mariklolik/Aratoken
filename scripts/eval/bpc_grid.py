import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from aratoken.metrics import per_doc_bpc
from aratoken.splits import load_texts
from aratoken.training import load_causal_lm


def find_checkpoint(run_dir: Path) -> tuple[Path | None, int | None]:
    model_dir = run_dir / "model"
    if (model_dir / "config.json").exists():
        return model_dir, None
    cks = sorted(model_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    if not cks:
        return None, None
    return cks[-1], int(cks[-1].name.split("-")[-1])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--grid-dir", type=Path, default=Path("outputs/grid"))
    p.add_argument("--manifest", type=Path, default=Path("outputs/grid/manifest.tsv"))
    p.add_argument("--split", type=Path, default=Path("data/fineweb2hq-ar/test"))
    p.add_argument("--limit", type=int, default=1500)
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", type=Path, default=Path("results/grid_bpc_with_init.json"))
    p.add_argument("--min-step", type=int, default=0, help="Skip runs whose latest checkpoint is below this step.")
    p.add_argument("--base-model", default="Qwen/Qwen3-0.6B-Base")
    p.add_argument("--only", default=None, help="Regex on the run name.")
    p.add_argument("--merge", action="store_true", help="Keep runs already scored in --out.")
    return p.parse_args()


def main() -> int:
    a = parse_args()
    texts = load_texts(a.split, a.limit, min_chars=200)
    with a.manifest.open() as fh:
        manifest = {r["name"]: r for r in csv.DictReader(fh, delimiter="\t")}
    prior = json.loads(a.out.read_text()) if a.merge and a.out.exists() else {}
    per_run, per_doc = dict(prior.get("per_run", {})), dict(prior.get("per_doc", {}))
    steps = dict(prior.get("checkpoint_steps", {}))

    base_key = "__base_" + a.base_model.split("/")[-1].lower().removesuffix("-base").replace("-", "_")
    targets = [] if base_key in per_run else [(base_key, a.base_model, a.base_model)]
    only = re.compile(a.only) if a.only else None
    for d in sorted(d for d in a.grid_dir.iterdir() if d.is_dir() and d.name != "logs"):
        if only and not only.search(d.name):
            continue
        ck, step = find_checkpoint(d)
        if ck is None or (a.min_step and step is not None and step < a.min_step):
            print(f"  skip {d.name}: checkpoint {step}")
            continue
        steps[d.name] = step
        targets.append((d.name, str(ck), manifest[d.name]["tokenizer"] if d.name in manifest else str(ck)))

    for name, model_path, tok_path in targets:
        tok = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
        model = load_causal_lm(model_path, a.device).eval()
        vals, _ = per_doc_bpc(model, tok, texts, a.device, a.max_len)
        per_run[name] = float(vals.mean())
        per_doc[name] = [round(float(v), 6) for v in vals]
        print(f"  {name:38} BPC {vals.mean():.4f}", flush=True)
        del model
        torch.cuda.empty_cache()

    groups: dict[str, list[float]] = defaultdict(list)
    for name, bpc in per_run.items():
        groups[name if name.startswith("__") else re.sub(r"_s\d+$", "", name)].append(bpc)
    agg = {k: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else None,
               "n_seeds": len(v), "values": v} for k, v in sorted(groups.items())}
    out = {"split": str(a.split), "n_docs": len(texts), "metric": "bits-per-character (lower is better)",
           "per_run": per_run, "by_config": agg, "per_doc": per_doc, "checkpoint_steps": steps}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for k, v in sorted(agg.items(), key=lambda kv: kv[1]["mean"]):
        print(f"{k:42} {v['mean']:8.4f} n={v['n_seeds']}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
