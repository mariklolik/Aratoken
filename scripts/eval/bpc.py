import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from aratoken.metrics import bootstrap_ci, per_doc_bpc
from aratoken.splits import load_texts
from aratoken.training import load_causal_lm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", action="append", required=True, metavar="TAG=MODEL[:TOKENIZER]")
    p.add_argument("--split", type=Path, default=Path("data/fineweb2hq-ar/test"))
    p.add_argument("--lang", choices=["ar", "en"], default="ar",
                   help="en scores WikiText-103 for the English retention check.")
    p.add_argument("--limit", type=int, default=2000)
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n-bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def english_texts(limit: int) -> list[str]:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    return [t for t in ds["text"] if len(t) >= 200][:limit]


def main() -> int:
    a = parse_args()
    texts = load_texts(a.split, a.limit, min_chars=200) if a.lang == "ar" else english_texts(a.limit)
    source = str(a.split) if a.lang == "ar" else "wikitext/wikitext-103-raw-v1:test"
    results = {}
    for spec in a.model:
        tag, _, path = spec.partition("=")
        model_path, _, tok_path = path.partition(":")
        tok = AutoTokenizer.from_pretrained(tok_path or model_path, trust_remote_code=True)
        model = load_causal_lm(model_path, a.device).eval()
        vals, totals = per_doc_bpc(model, tok, texts, a.device, a.max_len)
        results[tag] = {"bpc": bootstrap_ci(vals, a.n_bootstrap, a.seed), **totals, "vocab_size": len(tok),
                        "model_path": model_path, "tokenizer_path": tok_path or model_path}
        b = results[tag]["bpc"]
        print(f"{tag:40} BPC {b['mean']:.4f} [{b['ci_lower']:.4f}, {b['ci_upper']:.4f}] "
              f"chars/token {totals['chars_per_token']:.2f}", flush=True)
        del model
        torch.cuda.empty_cache()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"source": source, "lang": a.lang, "n_docs": len(texts), "max_len": a.max_len,
                                 "seed": a.seed, "results": results}, indent=2), encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
