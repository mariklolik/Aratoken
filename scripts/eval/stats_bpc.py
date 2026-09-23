import argparse
import json
import re
from pathlib import Path

import numpy as np

from aratoken.stats import holm, paired_bootstrap


def pooled_over_seeds(per_doc: dict[str, list[float]], skip: str = "") -> dict[str, np.ndarray]:
    groups: dict[str, list[np.ndarray]] = {}
    for name, vals in per_doc.items():
        if name != skip:
            groups.setdefault(re.sub(r"_s\d+$", "", name), []).append(np.asarray(vals, float))
    return {k: np.mean(np.stack(v), axis=0) for k, v in groups.items()}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bpc", type=Path, default=Path("results/grid_bpc_with_init.json"))
    p.add_argument("--baseline", default="__base_qwen3_0.6b")
    p.add_argument("--contrast", action="append", default=[], metavar="A:B",
                   help="Also test configuration A against B, paired over documents.")
    p.add_argument("--n-bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("results/stats.json"))
    a = p.parse_args()
    per_doc = json.loads(a.bpc.read_text())["per_doc"]
    base = np.asarray(per_doc[a.baseline], dtype=float)
    seeds = {}
    for name in per_doc:
        if name != a.baseline:
            cfg = re.sub(r"_s\d+$", "", name)
            seeds[cfg] = seeds.get(cfg, 0) + 1

    comparisons = {}
    for cfg, v in pooled_over_seeds(per_doc, skip=a.baseline).items():
        if len(v) != len(base):
            continue
        c = paired_bootstrap(v, base, a.n_bootstrap, a.seed)
        comparisons[cfg] = {**c, "n_seeds": seeds[cfg], "practically_meaningful": abs(c["mean_diff"]) >= 0.005}
    for k, c in holm({k: v["p_value"] for k, v in comparisons.items()}).items():
        comparisons[k].update(c)

    groups = pooled_over_seeds({k: v for k, v in per_doc.items() if k.startswith("factorial_")})
    factorial = {}
    for lr in sorted({k.split("_", 2)[2] for k in groups}):
        if f"factorial_preserve_{lr}" in groups and f"factorial_unify_{lr}" in groups:
            factorial[f"preserve_minus_unify@{lr}"] = paired_bootstrap(
                groups[f"factorial_preserve_{lr}"], groups[f"factorial_unify_{lr}"], a.n_bootstrap, a.seed)

    everything = pooled_over_seeds(per_doc)
    contrasts = {}
    for spec in a.contrast:
        left, right = spec.split(":", 1)
        contrasts[f"{left}_minus_{right}"] = paired_bootstrap(everything[left], everything[right],
                                                              a.n_bootstrap, a.seed)

    out = {"source": str(a.bpc), "baseline": a.baseline,
           "metric": "bits-per-character (lower is better); negative mean_diff favours the configuration",
           "correction": "Holm-Bonferroni across all configurations vs baseline",
           "comparisons": comparisons, "factorial_contrasts": factorial, "pairwise_contrasts": contrasts}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for k, c in sorted(comparisons.items(), key=lambda kv: kv[1]["mean_diff"]):
        print(f"{k:38} {c['mean_diff']:+.4f} [{c['ci_lower']:+.4f},{c['ci_upper']:+.4f}] p_holm={c['p_holm']:.4f}")
    for k, c in {**factorial, **contrasts}.items():
        print(f"{k:44} {c['mean_diff']:+.4f} p={c['p_value']:.4f}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
