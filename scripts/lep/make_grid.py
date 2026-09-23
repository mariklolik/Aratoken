import argparse
import csv
from pathlib import Path

QWEN06 = "Qwen/Qwen3-0.6B-Base"
QWEN17 = "Qwen/Qwen3-1.7B-Base"
TOKENIZER = {"preserve": "tokenizers/extended/fw2hq-alif-preserve-150k",
             "unify": "tokenizers/extended/fw2hq-alif-unified-150k"}
DATA = {"preserve": "data/pretokenized/fw2hq-alif-preserve",
        "unify": "data/pretokenized/fw2hq-alif-unified",
        "qwen": "data/pretokenized/fw2hq-qwen"}
LAYER_SETS = {
    "L27": list(range(27, 28)),
    "L24_27": list(range(24, 28)),
    "L18_27": list(range(18, 28)),
    "L9_27": list(range(9, 28)),
    "all": list(range(28)),
}
TOP4 = LAYER_SETS["L24_27"]
COLUMNS = ["arm", "name", "mode", "tokenizer", "model", "data", "lr", "layers", "seed", "max_steps", "overrides"]


def model(alif: str, init: str = "mean", size: str = "0.6b") -> str:
    return f"models/qwen3-{size}-fw2hq-alif-{alif}-150k-{init}"


def run(arm, name, mode, tokenizer, model_path, data, layers, seed, max_steps, lr=2e-4, overrides=""):
    return dict(arm=arm, name=f"{name}_s{seed}", mode=mode, tokenizer=tokenizer, model=model_path,
                data=data, lr=lr, layers=str(layers).replace(" ", ""), seed=seed,
                max_steps=max_steps, overrides=overrides)


def lep(arm, name, seed, layers=TOP4, alif="preserve", init="mean", lr=2e-4, overrides=""):
    return run(arm, name, "lep", TOKENIZER[alif], model(alif, init), DATA[alif], layers, seed, 2000, lr, overrides)


def rows(seeds: list[int]) -> list[dict]:
    out = []
    for s in seeds:
        for alif in ("preserve", "unify"):
            for tag, lr in (("lr8e-5", 8e-5), ("lr2e-4", 2e-4)):
                out.append(lep("factorial", f"factorial_{alif}_{tag}", s, alif=alif, lr=lr))
        for tag, layers in LAYER_SETS.items():
            out.append(lep("trainable", f"trainable_{tag}", s, layers=layers))
        for init in ("random", "calib", "wechsel", "focus", "hybrid"):
            out.append(lep("init", f"init_{init}", s, init=init))
        out.append(lep("baseline", "baseline_embonly", s, layers=[]))
        out.append(lep("baseline", "baseline_adapter", s, layers=[], overrides="lora_rank=16"))
        out.append(run("baseline", "baseline_cpt_origtok", "cpt", QWEN06, QWEN06, DATA["qwen"], TOP4, s, 2000))
        out.append(run("baseline", "baseline_cpt_equaltext", "cpt", QWEN06, QWEN06, DATA["qwen"], TOP4, s, 3584))
    scale = "training.per_device_train_batch_size=8 training.gradient_accumulation_steps=12"
    out.append(run("scale", "scale17hyb_lep", "lep", TOKENIZER["preserve"], model("preserve", "hybrid", "1.7b"),
                   DATA["preserve"], TOP4, 1, 2000, overrides=scale))
    out.append(run("scale", "scale17full_cpt", "cpt", QWEN17, QWEN17, DATA["qwen"], TOP4, 1, 2000, overrides=scale))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("outputs/grid/manifest.tsv"))
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    a = p.parse_args()
    data = rows(a.seeds)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(data)
    print(f"wrote {a.out} with {len(data)} runs\nsbatch --array=1-{len(data)} slurm/train.sbatch")


if __name__ == "__main__":
    main()
