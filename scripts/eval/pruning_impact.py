import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import sentencepiece as spm

from aratoken.normalizer import paper_normalizer
from aratoken.paths import PRUNING_LADDER
from aratoken.splits import load_texts

ARABIC = re.compile(r"[؀-ۿ]")
BANDS = {
    "top-1k": lambda r: r < 1000,
    "1k-10k": lambda r: 1000 <= r < 10000,
    "10k-100k": lambda r: 10000 <= r < 100000,
    "rare / unseen": lambda r: r >= 100000,
}


def pieces(sp) -> set[str]:
    return {sp.id_to_piece(i).replace("▁", "") for i in range(sp.get_piece_size())}


def dropped_pieces(full, pruned) -> dict:
    dropped = [p for p in pieces(full) - pieces(pruned) if p and ARABIC.search(p)]
    lens = np.array([len(p) for p in dropped])
    return {"n_dropped": len(dropped), "mean_len": float(lens.mean()),
            "median_len": float(np.median(lens)), "frac_len_ge5": float((lens >= 5).mean()),
            "examples_long": sorted(p for p in dropped if len(p) >= 8)[:12]}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--freq-docs", type=int, default=40000)
    p.add_argument("--test-words", type=int, default=300000)
    p.add_argument("--out", type=Path, default=Path("results/pruning_impact.json"))
    a = p.parse_args()
    norm = paper_normalizer(preserve_alif=False)
    sps = {name: spm.SentencePieceProcessor(model_file=path)
           for name, path in PRUNING_LADDER if Path(path).exists()}
    if "full" not in sps:
        raise SystemExit("need the unpruned model as reference")

    out: dict = {"dropped": {name: dropped_pieces(sps["full"], sps[name])
                             for name in ("99%", "95%") if name in sps},
                 "fertility_by_frequency_band": {}}
    freq: Counter = Counter()
    for t in load_texts(Path("data/fineweb2hq-ar/lm-train"), a.freq_docs):
        freq.update(norm.normalize_str(t).split())
    rank = {w: i for i, (w, _) in enumerate(freq.most_common())}
    words: list[str] = []
    for t in load_texts(Path("data/fineweb2hq-ar/test")):
        words.extend(norm.normalize_str(t).split())
        if len(words) >= a.test_words:
            break
    bucketed: dict[str, list[str]] = {k: [] for k in BANDS}
    for w in words[:a.test_words]:
        r = rank.get(w, 10**9)
        bucketed[next(b for b, test in BANDS.items() if test(r))].append(w)

    for band, ws in bucketed.items():
        if not ws:
            continue
        row = {name: sum(len(sp.encode(w, out_type=int)) for w in ws) / len(ws) for name, sp in sps.items()}
        if "95%" in row:
            row["pct_increase_full_to_95pct"] = 100 * (row["95%"] / row["full"] - 1)
        out["fertility_by_frequency_band"][band] = {"n_words": len(ws), **row}
        print(f"{band:16}{len(ws):>9,}" + "".join(f"{row[n]:>10.3f}" for n in sps))

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
