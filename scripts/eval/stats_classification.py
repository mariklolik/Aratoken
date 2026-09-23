import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from aratoken.stats import holm, paired_bootstrap_metric


def macro_f1(labels):
    return lambda t, p: f1_score(t, p, labels=labels, average="macro", zero_division=0)


def load_arm(d: Path) -> dict[tuple[str, int], tuple[np.ndarray, np.ndarray, list]]:
    acc: dict[tuple[str, int], list] = defaultdict(list)
    for f in sorted(d.glob("*.json")):
        r = json.loads(f.read_text())
        acc[(r["dataset"], r["shots"])].append(r)
    out = {}
    for key, recs in acc.items():
        recs.sort(key=lambda r: r["seed"])
        y_true = np.concatenate([np.asarray(r["y_true"], dtype=object) for r in recs])
        y_pred = np.concatenate([np.asarray(r["y_pred"], dtype=object) for r in recs])
        out[key] = (y_true, y_pred, recs[0]["labels"])
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--preds", type=Path, default=Path("results/downstream_preds_mcq"))
    p.add_argument("--baseline", default="qwen3_0_6b_base")
    p.add_argument("--n-bootstrap", type=int, default=2000,
                   help="Each resample recomputes Macro F1, so this is far "
                        "more expensive than the BPC bootstrap.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("results/stats_downstream.json"))
    return p.parse_args()


def main() -> int:
    a = parse_args()
    root = a.preds
    arms = {d.name: load_arm(d) for d in sorted(root.iterdir()) if d.is_dir()}
    if a.baseline not in arms:
        raise SystemExit(f"baseline {a.baseline} not under {root} "
                         f"(have {sorted(arms)})")
    base = arms[a.baseline]

    comparisons: dict[str, dict] = {}
    for arm, per_key in arms.items():
        if arm == a.baseline:
            continue
        for key, (yt, yp, labels) in per_key.items():
            if key not in base:
                continue
            bt, bp, _ = base[key]
            if len(bt) != len(yt) or not np.array_equal(bt, yt):
                print(f"  ! {arm} {key}: item sets differ; skipping")
                continue
            name = f"{arm}|{key[0]}|{key[1]}shot"
            comparisons[name] = paired_bootstrap_metric(
                yt, yp, bp, macro_f1(labels), a.n_bootstrap, a.seed)
            print(f"  {name:52} d={comparisons[name]['delta_macro_f1']:+.4f} "
                  f"p={comparisons[name]['p_value']:.4f}", flush=True)

    adj = holm({k: v["p_value"] for k, v in comparisons.items()})
    for k, v in comparisons.items():
        v.update(adj[k])

    out = {"baseline": a.baseline, "statistic": "macro F1 difference",
           "resampling": "paired bootstrap over evaluation items, seeds pooled",
           "correction": "Holm-Bonferroni across the benchmark family",
           "comparisons": comparisons}
    outp = a.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    n_sig = sum(v["reject_at_0.05"] for v in comparisons.values())
    print(f"\n{n_sig}/{len(comparisons)} comparisons survive Holm-Bonferroni")
    print(f"wrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
