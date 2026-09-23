from __future__ import annotations

import hashlib
import json
from pathlib import Path

HASH_MOD = 1000
BUCKETS = {
    "lm-train": range(0, 600),
    "tok-train": range(600, 900),
    "dev": range(900, 950),
    "test": range(950, 1000),
}


def split_of(doc_id: str) -> str:
    digest = hashlib.blake2b(doc_id.encode("utf-8"), digest_size=8).digest()
    bucket = int.from_bytes(digest, "big") % HASH_MOD
    for name, rng in BUCKETS.items():
        if bucket in rng:
            return name
    raise AssertionError(f"unreachable bucket {bucket}")


def iter_records(split_dir: Path):
    shards = sorted(Path(split_dir).glob("*.jsonl"))
    if not shards:
        raise SystemExit(f"no .jsonl shards under {split_dir}; run scripts/data/build_splits.py first")
    for shard in shards:
        with shard.open(encoding="utf-8") as fh:
            for line in fh:
                yield json.loads(line)


def load_texts(split_dir: Path, limit: int | None = None, min_chars: int = 0,
               q_max: float | None = None, q_min: float | None = None) -> list[str]:
    texts: list[str] = []
    for rec in iter_records(split_dir):
        text = rec.get("text") or ""
        if len(text) < min_chars:
            continue
        q = rec.get("quality_score")
        if q is not None and ((q_max is not None and q > q_max) or
                              (q_min is not None and q < q_min)):
            continue
        texts.append(text)
        if limit and len(texts) >= limit:
            break
    return texts
