import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--grid-dir", type=Path, default=Path("outputs/grid"))
    p.add_argument("--only", default=r"^(init_|factorial_preserve_lr2e-4)")
    p.add_argument("--out", type=Path, default=Path("results/init_start_loss.json"))
    return p.parse_args()


def main() -> int:
    a = parse_args()
    grid = a.grid_dir
    keep = re.compile(a.only)
    by_config: dict[str, list[float]] = defaultdict(list)
    at_500: dict[str, list[float]] = defaultdict(list)
    for run in sorted(d for d in grid.iterdir() if d.is_dir()):
        if not keep.search(run.name):
            continue
        states = sorted(run.glob("model/checkpoint-*/trainer_state.json"),
                        key=lambda p: int(p.parent.name.split("-")[-1]))
        if not states:
            continue
        hist = json.loads(states[-1].read_text())["log_history"]
        ev = {e["step"]: e["eval_loss"] for e in hist if "eval_loss" in e}
        if not ev:
            continue
        cfg = re.sub(r"_s\d+$", "", run.name)
        by_config[cfg].append(ev[min(ev)])
        if 500 in ev:
            at_500[cfg].append(ev[500])

    out = {
        "metric": "dev cross-entropy per token; comparable across these arms "
                  "because they share a tokenizer",
        "first_step": {k: {"mean": float(np.mean(v)), "n_seeds": len(v),
                           "values": v} for k, v in sorted(by_config.items())},
        "step_500": {k: {"mean": float(np.mean(v)), "n_seeds": len(v)}
                     for k, v in sorted(at_500.items())},
    }
    outp = a.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"{'configuration':32} {'step 1':>8} {'step 500':>9}")
    for k, v in sorted(out["first_step"].items(), key=lambda kv: kv[1]["mean"]):
        s5 = out["step_500"].get(k, {}).get("mean")
        print(f"{k:32} {v['mean']:8.4f} {('%9.4f' % s5) if s5 else '        -'}")
    print(f"\nwrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
