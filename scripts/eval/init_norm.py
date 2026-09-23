import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen3-0.6B-Base"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--base", default=BASE)
    p.add_argument("--limit", type=int, default=20000)
    p.add_argument("--out", type=Path, default=Path("results/init_norm_shrinkage.json"))
    return p.parse_args()


@torch.no_grad()
def main() -> int:
    a = parse_args()
    base_tok = AutoTokenizer.from_pretrained(a.base, trust_remote_code=True)
    new_tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.base, torch_dtype=torch.float32, trust_remote_code=True)

    emb = model.get_input_embeddings().weight.data
    old_n = emb.shape[0]
    mu = emb.mean(0)
    donor = (emb - mu).norm(dim=1)

    added = {i: t for t, i in new_tok.get_vocab().items() if i >= old_n}
    by_k: dict[int, list[float]] = defaultdict(list)
    for i in sorted(added)[:a.limit]:
        ids = base_tok(added[i], add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        by_k[len(ids)].append(float((emb[torch.tensor(ids)].mean(0) - mu).norm()))

    donor_mean = float(donor.mean())
    per_k = {
        str(k): {"n_tokens": len(v), "radius": float(np.mean(v)),
                 "ratio_to_donor": float(np.mean(v)) / donor_mean,
                 "uncorrelated_prediction": k ** -0.5}
        for k, v in sorted(by_k.items()) if len(v) >= 20
    }
    allv = [x for v in by_k.values() for x in v]
    out = {
        "base_model": a.base, "tokenizer": a.tokenizer,
        "donor_radius_mean": donor_mean,
        "donor_radius_std": float(donor.std()),
        "new_tokens_measured": len(allv),
        "subtokens_per_token_mean": float(np.mean(
            [k for k, v in by_k.items() for _ in v])),
        "mean_init_radius": float(np.mean(allv)),
        "mean_init_ratio_to_donor": float(np.mean(allv)) / donor_mean,
        "tied_embeddings": bool(model.config.tie_word_embeddings),
        "by_subtoken_count": per_k,
    }
    outp = a.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"donor radius {donor_mean:.4f}; mean-init radius "
          f"{out['mean_init_radius']:.4f} "
          f"({out['mean_init_ratio_to_donor']:.3f} of donor)")
    print(f"{'k':>3} {'n':>7} {'radius':>8} {'ratio':>7} {'1/sqrt(k)':>10}")
    for k, v in per_k.items():
        print(f"{k:>3} {v['n_tokens']:7d} {v['radius']:8.4f} "
              f"{v['ratio_to_donor']:7.3f} {v['uncorrelated_prediction']:10.3f}")
    print(f"\nwrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
