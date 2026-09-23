import argparse
from pathlib import Path

from aratoken.init import METHODS, build


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=METHODS,
                   help="calib is the moment-matched variant of mean sub-token initialization.")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--base", default="Qwen/Qwen3-0.6B-Base")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.5, help="Weight of the compositional estimate in hybrid.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    build(a.method, a.tokenizer, a.out, a.base, a.topk, a.seed, a.alpha)
