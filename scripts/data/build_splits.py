import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

from aratoken.splits import BUCKETS, split_of

REPO = "epfml/FineWeb2-HQ"
CONFIG = "arb_Arab"
COLUMNS = ["id", "text", "quality_score", "language_score", "dump", "url"]

DEFAULT_BUDGET_CHARS = {
    "lm-train": 2_000_000_000,
    "tok-train": 480_000_000,
    "dev": 8_000_000,
    "test": 20_000_000,
}


@dataclass
class SplitWriter:
    name: str
    out_dir: Path
    budget_chars: int
    shard_chars: int = 256_000_000
    docs: int = 0
    chars: int = 0
    _shard_idx: int = 0
    _shard_chars: int = 0
    _fh: object = None
    _lock: Lock = field(default_factory=Lock)

    def __post_init__(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)

    @property
    def full(self) -> bool:
        return self.chars >= self.budget_chars

    def _rotate(self) -> None:
        if self._fh is not None:
            self._fh.close()
        path = self.out_dir / f"shard_{self._shard_idx:05d}.jsonl"
        self._fh = path.open("w", encoding="utf-8")
        self._shard_idx += 1
        self._shard_chars = 0

    def write(self, rec: dict) -> bool:
        with self._lock:
            if self.chars >= self.budget_chars:
                return False
            if self._fh is None or self._shard_chars >= self.shard_chars:
                self._rotate()
            self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n = len(rec["text"])
            self.docs += 1
            self.chars += n
            self._shard_chars += n
            return True

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None


def iter_shard(fs: HfFileSystem, path: str, min_chars: int, min_quality: float,
               keep_ids: set[str] | None):
    with fs.open(path, "rb") as fh:
        pf = pq.ParquetFile(fh)
        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg, columns=COLUMNS)
            for rec in tbl.to_pylist():
                if keep_ids is not None and rec["id"] not in keep_ids:
                    continue
                text = rec.get("text") or ""
                if len(text) < min_chars:
                    continue
                q = rec.get("quality_score")
                if min_quality > 0 and (q is None or q < min_quality):
                    continue
                yield {
                    "id": rec["id"],
                    "text": text,
                    "quality_score": q,
                    "dump": rec.get("dump"),
                    "url": rec.get("url"),
                }


def restore_order(split_dir: Path, order: dict[str, int]) -> None:
    shards = sorted(split_dir.glob("*.jsonl"))
    lines = [line for shard in shards for line in shard.read_text(encoding="utf-8").splitlines()]
    lines.sort(key=lambda line: order[json.loads(line)["id"]])
    for shard in shards:
        shard.unlink()
    (split_dir / "shard_00000.jsonl").write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data/fineweb2hq-ar"))
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--min-chars", type=int, default=200,
                   help="Drop documents shorter than this.")
    p.add_argument("--min-quality", type=float, default=0.0,
                   help="Drop documents below this FineWeb2-HQ quality_score. "
                        "The corpus is already the top-10%% of FineWeb2, so 0 "
                        "keeps everything.")
    p.add_argument("--shard-stride", type=int, default=3,
                   help="Read every Nth parquet shard rather than a prefix, so "
                        "the sample spans the whole crawl range instead of "
                        "over-representing the earliest dumps.")
    p.add_argument("--ids", type=Path, nargs="+", default=None,
                   help="Keep only these document ids, e.g. data/splits/test_ids.txt, "
                        "to rebuild the released test and dev splits exactly.")
    for name, default in DEFAULT_BUDGET_CHARS.items():
        p.add_argument(f"--budget-{name}", type=int, default=default,
                       dest=f"budget_{name.replace('-', '_')}")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    budgets = {name: getattr(args, f"budget_{name.replace('-', '_')}")
               for name in BUCKETS}

    keep_ids = ({line.strip() for f in args.ids for line in f.read_text().splitlines() if line.strip()}
                if args.ids else None)
    fs = HfFileSystem()
    all_shards = sorted(
        f for f in fs.ls(f"datasets/{REPO}/{CONFIG}", detail=False)
        if f.endswith(".parquet")
    )
    shards = all_shards[::args.shard_stride]
    print(f"{len(all_shards)} shards available, reading {len(shards)} "
          f"(stride {args.shard_stride}) with {args.workers} workers")
    for name, b in budgets.items():
        print(f"  budget {name:10} {b/1e6:9.1f} Mchar")

    writers = {name: SplitWriter(name, args.out / name, budgets[name])
               for name in BUCKETS}
    seen_ids: set[str] = set()
    seen_lock = Lock()
    t0 = time.time()
    stop = False

    def consume(path: str) -> tuple[int, int]:
        nonlocal stop
        kept = skipped = 0
        for rec in iter_shard(fs, path, args.min_chars, args.min_quality, keep_ids):
            if stop:
                break
            doc_id = rec["id"]
            with seen_lock:
                if doc_id in seen_ids:
                    skipped += 1
                    continue
                seen_ids.add(doc_id)
            w = writers[split_of(doc_id)]
            if w.full:
                if all(x.full for x in writers.values()):
                    stop = True
                    break
                continue
            w.write(rec)
            kept += 1
        return kept, skipped

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(consume, s) for s in shards]
        done = 0
        for fut in futures:
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"  shard failed: {exc}", file=sys.stderr)
            done += 1
            if done % 5 == 0 or stop:
                el = time.time() - t0
                tot = sum(w.chars for w in writers.values())
                print(f"  [{done}/{len(shards)}] {tot/1e6:8.1f} Mchar "
                      f"in {el:5.0f}s ({tot/1e6/max(el,1):.1f} Mchar/s)")
            if stop:
                break

    for w in writers.values():
        w.close()
    if args.ids:
        order = {doc_id: i for f in args.ids for i, doc_id in enumerate(f.read_text().split())}
        for w in writers.values():
            restore_order(w.out_dir, order)

    manifest = {
        "source": {"repo": REPO, "config": CONFIG,
                   "shards_read": len(shards), "shards_total": len(all_shards),
                   "shard_stride": args.shard_stride},
        "split_rule": {
            "hash": "blake2b(id, digest_size=8) % 1000",
            "buckets": {k: [v.start, v.stop] for k, v in BUCKETS.items()},
            "note": "Disjoint by construction: each document id maps to exactly "
                    "one bucket, independent of read order or budgets.",
        },
        "filters": {"min_chars": args.min_chars, "min_quality": args.min_quality},
        "splits": {n: {"docs": w.docs, "chars": w.chars,
                       "budget_chars": w.budget_chars,
                       "approx_tokens": int(w.chars / 3.9)}
                   for n, w in writers.items()},
        "unique_ids_seen": len(seen_ids),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== splits ===")
    for n, s in manifest["splits"].items():
        print(f"  {n:10} {s['docs']:9,} docs  {s['chars']/1e6:9.1f} Mchar  "
              f"~{s['approx_tokens']/1e6:7.1f} Mtok")
    print(f"manifest -> {args.out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
