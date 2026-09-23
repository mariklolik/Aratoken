import argparse
import json
import re
from itertools import islice
from pathlib import Path

from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

from aratoken.splits import iter_records

N_GRAM = 5
NUM_PERM = 112
BANDS = 14
THRESHOLD = 0.72
WORD = re.compile(r"\w+", re.UNICODE)


def shingles(text: str) -> set[bytes]:
    words = WORD.findall(text)
    if len(words) < N_GRAM:
        return {" ".join(words).encode("utf-8")} if words else set()
    return {" ".join(words[i:i + N_GRAM]).encode("utf-8")
            for i in range(len(words) - N_GRAM + 1)}


def signature(text: str) -> MinHash:
    m = MinHash(num_perm=NUM_PERM)
    m.update_batch(list(shingles(text)))
    return m


def iter_docs(split_dir: Path, limit: int | None):
    docs = (rec for rec in iter_records(split_dir) if rec.get("text"))
    return islice(docs, limit)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/fineweb2hq-ar"))
    p.add_argument("--test-split", default="test")
    p.add_argument("--train-splits", nargs="+", default=["tok-train", "lm-train"])
    p.add_argument("--out", type=Path, default=Path("results/minhash_leakage.json"))
    p.add_argument("--max-train-docs", type=int, default=None)
    p.add_argument("--max-test-docs", type=int, default=None)
    return p.parse_args()


def main() -> int:
    a = parse_args()
    root = a.root

    lsh = MinHashLSH(threshold=THRESHOLD, num_perm=NUM_PERM,
                     params=(BANDS, NUM_PERM // BANDS))
    indexed = {name: 0 for name in a.train_splits}
    with lsh.insertion_session() as session:
        for name in a.train_splits:
            for rec in tqdm(iter_docs(root / name, a.max_train_docs), desc=f"index {name}"):
                session.insert(f"{name}/{rec['id']}", signature(rec["text"]),
                               check_duplication=False)
                indexed[name] += 1

    hits, n_test, exact = [], 0, 0
    seen_exact: set[int] = set()
    for rec in tqdm(iter_docs(root / a.test_split, a.max_test_docs), desc="query test"):
        n_test += 1
        h = hash(rec["text"])
        if h in seen_exact:
            exact += 1
        seen_exact.add(h)
        matches = lsh.query(signature(rec["text"]))
        if matches:
            hits.append({"test_id": rec["id"], "n_matches": len(matches),
                         "example": matches[0]})

    out = {
        "config": {"n_gram": N_GRAM, "num_perm": NUM_PERM, "bands": BANDS,
                   "rows_per_band": NUM_PERM // BANDS, "threshold": THRESHOLD,
                   "reference": "RefinedWeb/FineWeb MinHash settings"},
        "indexed_train_docs": indexed,
        "test_docs": n_test,
        "near_duplicate_docs": len(hits),
        "near_duplicate_rate": len(hits) / n_test if n_test else 0.0,
        "within_test_exact_duplicates": exact,
        "examples": hits[:20],
    }
    outp = a.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{len(hits)}/{n_test} test documents ({100 * out['near_duplicate_rate']:.3f}%) "
          f"have a near duplicate at Jaccard >= {THRESHOLD} in "
          f"{sum(indexed.values())} training documents")
    print(f"wrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
