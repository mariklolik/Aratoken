import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score


def chance_floor(y: list, labels: list, rng) -> dict:
    counts = Counter(y)
    prior = np.array([counts[lbl] for lbl in labels], dtype=float)
    prior /= prior.sum()
    return {
        "majority_class": float(f1_score(y, [counts.most_common(1)[0][0]] * len(y), labels=labels, average="macro")),
        "label_prior": float(np.mean([f1_score(y, list(rng.choice(labels, size=len(y), p=prior)),
                                               labels=labels, average="macro") for _ in range(20)])),
    }


def summary(values: list[float]) -> dict:
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "n_seeds": len(values), "values": values}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--preds", type=Path, default=Path("results/downstream_preds_mcq"))
    p.add_argument("--out", type=Path, default=Path("results/downstream.json"))
    a = p.parse_args()
    dumps: dict[tuple, dict[int, dict]] = defaultdict(dict)
    for arm_dir in sorted(d for d in a.preds.iterdir() if d.is_dir()):
        for f in sorted(arm_dir.glob("*.json")):
            r = json.loads(f.read_text())
            dumps[(arm_dir.name, r["dataset"], int(r["shots"]))][int(r["seed"])] = r

    rng = np.random.default_rng(0)
    floors: dict[tuple, dict] = {}
    out: dict[str, dict] = {}
    for (tag, dataset, shots), by_seed in dumps.items():
        runs = [r for _, r in sorted(by_seed.items())]
        if (dataset, shots) not in floors:
            floors[(dataset, shots)] = chance_floor(runs[0]["y_true"], runs[0]["labels"], rng)
        out.setdefault(tag, {}).setdefault(dataset, {})[f"{shots}shot"] = {
            "macro_f1": summary([float(f1_score(r["y_true"], r["y_pred"], labels=r["labels"], average="macro"))
                                 for r in runs]),
            "balanced_acc": summary([float(balanced_accuracy_score(r["y_true"], r["y_pred"])) for r in runs]),
            "seeds": sorted(by_seed),
            "label_usage": {f"seed{s}": {"labels_available": len(r["labels"]),
                                         "labels_ever_predicted": len(Counter(r["y_pred"])),
                                         "counts": dict(Counter(r["y_pred"]).most_common())}
                            for s, r in sorted(by_seed.items())},
            "chance": floors[(dataset, shots)],
        }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    for tag, per_dataset in out.items():
        for dataset, per_shot in per_dataset.items():
            print(f"{tag:20} {dataset:20} " + "  ".join(
                f"{s}: {v['macro_f1']['mean']:.3f}" for s, v in sorted(per_shot.items())))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
