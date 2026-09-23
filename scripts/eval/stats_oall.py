import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from aratoken.stats import accuracy, holm, paired_bootstrap_metric


def load(d: Path) -> dict[str, dict[str, np.ndarray]]:
    arms: dict[str, dict[str, np.ndarray]] = {}
    for f in sorted(d.glob("*.json")):
        if f.stem.endswith("_summary"):
            continue
        tag, task = f.stem.split("_", 1)
        r = json.loads(f.read_text())
        arms.setdefault(tag, {})[task] = (np.array(r["y_true"]),
                                          np.array(r["y_pred"]))
    return arms


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=Path, default=Path("results/oall/mcq"))
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("results/stats_oall.json"))
    a = p.parse_args()

    arms = load(a.dir)
    tasks = sorted(set.intersection(*(set(v) for v in arms.values())))

    out: dict[str, dict] = {"per_task": {}, "pooled": {}, "accuracy": {}}
    for tag, per_task in arms.items():
        out["accuracy"][tag] = {t: float(np.mean(per_task[t][0] == per_task[t][1]))
                                for t in tasks}
        out["accuracy"][tag]["macro"] = float(
            np.mean([out["accuracy"][tag][t] for t in tasks]))
        yt = np.concatenate([per_task[t][0] for t in tasks])
        yp = np.concatenate([per_task[t][1] for t in tasks])
        out["accuracy"][tag]["pooled"] = float(np.mean(yt == yp))

    pvals = {}
    for x, y in combinations(sorted(arms), 2):
        yt = np.concatenate([arms[x][t][0] for t in tasks])
        xp = np.concatenate([arms[x][t][1] for t in tasks])
        yp = np.concatenate([arms[y][t][1] for t in tasks])
        r = paired_bootstrap_metric(yt, xp, yp, accuracy, a.n_bootstrap, a.seed,
                             name="delta_accuracy")
        out["pooled"][f"{x}:{y}"] = r
        pvals[f"{x}:{y}"] = r["p_value"]
        for t in tasks:
            tt, tx = arms[x][t]
            _, ty = arms[y][t]
            out["per_task"].setdefault(t, {})[f"{x}:{y}"] = paired_bootstrap_metric(
                tt, tx, ty, accuracy, a.n_bootstrap, a.seed,
                name="delta_accuracy")
    out["holm_pooled"] = holm(pvals)

    outp = a.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {outp}\n")
    for tag, acc in sorted(out["accuracy"].items()):
        cells = "  ".join(f"{t} {acc[t]:.4f}" for t in tasks)
        print(f"{tag:6} macro {acc['macro']:.4f} pooled {acc['pooled']:.4f}   {cells}")
    print()
    for k, v in out["pooled"].items():
        h = out["holm_pooled"][k]
        print(f"{k:12} delta {v['delta_accuracy']:+.4f} "
              f"[{v['ci_lower']:+.4f}, {v['ci_upper']:+.4f}] "
              f"p {v['p_value']:.4f} holm {h['p_holm']:.4f} n={v['n_items']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
