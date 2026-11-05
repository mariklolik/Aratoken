import argparse
import csv
import json
from pathlib import Path

from scripts.lep.make_grid import COLUMNS, DATA, QWEN06, QWEN17, TOKENIZER, TOP4, run

INITIALIZERS = {
    "random": ("random", .5),
    "mean": ("mean", 1.),
    "calib": ("calib", .5),
    "analogical": ("hybrid", 0.),
    "hybrid": ("hybrid", .5),
    "alpha025": ("hybrid", .25),
    "alpha075": ("hybrid", .75),
}
INITIALIZER_ROOT = "outputs/rebuttal-initializers"
OVERRIDES = (
    "training.per_device_train_batch_size=8 training.gradient_accumulation_steps=12 "
    "training.per_device_eval_batch_size=4 training.load_best_model_at_end=false "
    "training.save_steps=1000 training.eval_steps=1000 training.save_total_limit=1 "
    "training.dataloader_num_workers=2 training.dataloader_drop_last=true training.report_to=none"
)


def rebuttal_rows(seeds: list[int]) -> list[dict]:
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 1:
        raise ValueError("seeds must be distinct positive integers")
    matrix = []
    for arm, size, base, steps in (("small", "0.6b", QWEN06, 2000),
                                    ("budget500m", "0.6b", QWEN06, 5087),
                                    ("scale17", "1.7b", QWEN17, 2000)):
        for seed in seeds:
            methods = list(INITIALIZERS) if arm == "small" else list(INITIALIZERS)[:5]
            for method in methods:
                matrix.append(run(arm, f"{arm}_{method}", "lep", TOKENIZER["preserve"],
                                  f"{INITIALIZER_ROOT}/{size}-{method}_s{seed}",
                                  DATA["preserve"], TOP4, seed, steps, overrides=OVERRIDES))
            fair = OVERRIDES + " train_embeddings=true"
            matrix.append(run(arm, f"{arm}_cpt", "cpt", base, base, DATA["qwen"],
                              TOP4, seed, steps, overrides=fair))
            matrix.append(run(arm, f"{arm}_lep_all", "cpt", TOKENIZER["preserve"],
                              f"{INITIALIZER_ROOT}/{size}-hybrid_s{seed}", DATA["preserve"],
                              list(range(28)), seed, steps, overrides=fair))
            matrix.append(run(arm, f"{arm}_cpt_all", "cpt", base, base, DATA["qwen"],
                              list(range(28)), seed, steps, overrides=fair))
    return matrix


def initializer_specs(matrix: list[dict]) -> dict:
    specs = {}
    for row in matrix:
        path = row["model"]
        if not path.startswith(INITIALIZER_ROOT + "/"):
            continue
        name, seed = Path(path).name.rsplit("_s", 1)
        size, initializer = name.split("-", 1)
        method, alpha = INITIALIZERS[initializer]
        specs[path] = dict(method=method, alpha=alpha, seed=int(seed),
                           base=QWEN06 if size == "0.6b" else QWEN17,
                           tokenizer=row["tokenizer"], out=path, topk=10)
    return specs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("outputs/rebuttal/manifest.tsv"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()
    matrix = rebuttal_rows(args.seeds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(matrix)
    args.out.with_suffix(".initializers.json").write_text(json.dumps(initializer_specs(matrix), indent=2))
    print(f"{len(matrix)} runs; {len(initializer_specs(matrix))} initial states; {args.out}")


if __name__ == "__main__":
    main()
