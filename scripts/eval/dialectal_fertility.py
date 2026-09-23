import argparse
import json
from pathlib import Path

import numpy as np

from aratoken.normalizer import paper_normalizer
from aratoken.paths import ARATOKEN, BASELINES
from aratoken.splits import load_texts

SYSTEMS = [(n, "sp", path, preserve) for n, (path, preserve) in ARATOKEN.items()] + [
    (n, "hf", BASELINES[n], True) for n in ("AraBERT", "Qwen3")]


def load_encoder(kind: str, path: str):
    if kind == "sp":
        import sentencepiece as spm
        if not Path(path).exists():
            return None, None
        sp = spm.SentencePieceProcessor(model_file=path)
        return (lambda t: sp.encode(t, out_type=int)), (sp.get_piece_size(), sp.unk_id())
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    return (lambda t: tok(t, add_special_tokens=False)["input_ids"]), (len(tok), getattr(tok, "unk_token_id", None))


def measure(enc, unk_id, texts, norm) -> dict:
    fert, unk_frac = [], []
    for t in texts:
        tt = norm.normalize_str(t)
        w = len(tt.split())
        ids = enc(tt) if w else []
        if not ids:
            continue
        fert.append(len(ids) / w)
        if unk_id is not None:
            unk_frac.append(sum(1 for i in ids if i == unk_id) / len(ids))
    return {"fertility_mean": float(np.mean(fert)), "fertility_std": float(np.std(fert, ddof=1)),
            "unk_rate": float(np.mean(unk_frac)) if unk_frac else 0.0, "n_docs": len(fert)}


def load_dialectal(limit: int) -> list[str]:
    from datasets import load_dataset
    ds = load_dataset("iabufarha/ar_sarcasm", split="test")
    return [t for t in ds["tweet"] if t and len(t.strip()) > 20][:limit]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=3000)
    p.add_argument("--low-quality-max", type=float, default=0.1145,
                   help="Bottom-quartile quality_score threshold of the test split.")
    p.add_argument("--out", type=Path, default=Path("results/dialectal_fertility.json"))
    a = p.parse_args()
    test = Path("data/fineweb2hq-ar/test")
    corpora = {"msa_test": load_texts(test, a.limit),
               "msa_low_quality_band": load_texts(test, a.limit, q_max=a.low_quality_max),
               "dialectal_arsarcasm": load_dialectal(a.limit)}
    results: dict[str, dict] = {}
    for name, kind, path, preserve in SYSTEMS:
        enc, meta = load_encoder(kind, path)
        if enc is None:
            print(f"  skip {name}")
            continue
        norm = paper_normalizer(preserve)
        results[name] = {"vocab_size": meta[0],
                         **{c: measure(enc, meta[1], texts, norm) for c, texts in corpora.items()}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"{'system':36}{'MSA':>9}{'low-q':>9}{'dialect':>9}")
    for n, r in results.items():
        print(f"{n:36}" + "".join(f"{r[c]['fertility_mean']:9.3f}" for c in corpora))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
