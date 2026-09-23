import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from aratoken.metrics import fertility_and_compression
from aratoken.normalizer import build_qwen_like_normalizer, config_from_spec, paper_normalizer
from aratoken.paths import ARATOKEN, BASELINES, EXTENDED
from aratoken.splits import load_texts


@dataclass
class Entry:
    name: str
    kind: str
    path: str
    own_input: str = "norm"
    note: str = ""


def default_registry() -> list[Entry]:
    entries = [Entry(n, "sp", path, "norm" if preserve else "norm_unified")
               for n, (path, preserve) in ARATOKEN.items()]
    entries += [Entry(n, "hf_local", path, "norm" if "preserve" in n else "norm_unified")
                for n, path in EXTENDED.items()]
    entries += [Entry(n, "hf_hub", path) for n, path in BASELINES.items()]
    return entries


def load_tokenizer(e: Entry):
    if e.kind in ("sp", "tok_json", "hf_local") and not Path(e.path).exists():
        return None, None, None
    if e.kind == "sp":
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=e.path)
        return (lambda t: sp.encode(t, out_type=int)), sp.get_piece_size(), sp.unk_id()
    if e.kind == "tok_json":
        from tokenizers import Tokenizer
        t = Tokenizer.from_file(e.path)
        return (lambda s: t.encode(s, add_special_tokens=False).ids), t.get_vocab_size(), t.token_to_id("<unk>")
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(e.path, trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  ! {e.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None, None, None
    return (lambda t: tok(t, add_special_tokens=False)["input_ids"]), len(tok), tok.unk_token_id


def own_normalizer(e: Entry):
    cfg = Path(e.path)
    cfg = (cfg.parent if cfg.is_file() else cfg) / "tokenizer_config.json"
    if not cfg.exists():
        return None
    return build_qwen_like_normalizer(config_from_spec(json.loads(cfg.read_text()).get("normalizer_spec", {})))


def morpheme_segmenter():
    try:
        from camel_tools.disambig.mle import MLEDisambiguator
        from camel_tools.tokenizers.morphological import MorphologicalTokenizer
    except ImportError:
        return None
    tok = MorphologicalTokenizer(MLEDisambiguator.pretrained(), scheme="d3tok", split=True)
    return lambda words: tok.tokenize(words)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--split", type=Path, default=Path("data/fineweb2hq-ar/test"))
    p.add_argument("--registry", type=Path, default=None,
                   help="JSON list of {name, kind, path, own_input}; configs/registry_algorithms.json "
                        "measures the algorithm grid with the same code path.")
    p.add_argument("--out", type=Path, default=Path("results/intrinsic_metrics.json"))
    p.add_argument("--limit", type=int, default=5000)
    p.add_argument("--n-bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quality-max", type=float, default=None)
    p.add_argument("--quality-min", type=float, default=None)
    p.add_argument("--report-morph", action="store_true",
                   help="Also report tokens per morpheme against the CAMeL Tools d3tok segmentation.")
    p.add_argument("--morph-docs", type=int, default=500)
    p.add_argument("--oov-docs", type=int, default=500)
    return p.parse_args()


def main() -> int:
    a = parse_args()
    texts_raw = load_texts(a.split, a.limit, q_max=a.quality_max, q_min=a.quality_min)
    norm_keep, norm_unify = paper_normalizer(True), paper_normalizer(False)
    inputs = {"raw": texts_raw,
              "norm": [norm_keep.normalize_str(t) for t in texts_raw],
              "norm_unified": [norm_unify.normalize_str(t) for t in texts_raw]}
    print(f"loaded {len(texts_raw)} docs from {a.split}")

    registry = ([Entry(**r) for r in json.loads(a.registry.read_text())] if a.registry
                else default_registry())
    morph_counts = None
    if a.report_morph:
        seg = morpheme_segmenter()
        if seg is None:
            print("  ! --report-morph needs camel_tools; skipping")
        else:
            morph_counts = [len(seg(t.split())) for t in inputs["norm"][:a.morph_docs]]

    results: dict[str, dict] = {}
    for e in registry:
        encode, vocab, unk_id = load_tokenizer(e)
        if encode is None:
            print(f"  skip {e.name} (unavailable)")
            continue
        print(f"  measuring {e.name} (vocab {vocab})")
        if e.own_input == "auto":
            n = own_normalizer(e)
            own = [n.normalize_str(t) for t in texts_raw] if n else inputs["norm"]
        else:
            own = inputs[e.own_input]
        results[e.name] = {
            "vocab_size": vocab, "note": e.note, "source": e.path, "own_input": e.own_input,
            "raw": fertility_and_compression(encode, inputs["raw"], a.n_bootstrap, a.seed),
            "normalized": fertility_and_compression(encode, inputs["norm"], a.n_bootstrap, a.seed),
            "own_norm": fertility_and_compression(encode, own, a.n_bootstrap, a.seed, unk_id, a.oov_docs),
        }
        if morph_counts is not None:
            toks = sum(len(encode(t)) for t in own[:len(morph_counts)])
            results[e.name]["tokens_per_morpheme"] = toks / sum(morph_counts)

    out = {"split": str(a.split), "n_docs": len(texts_raw), "seed": a.seed,
           "fertility_definition": "tokens per whitespace-delimited word (str.split())",
           "conventions": {"macro": "per-document mean, bootstrap CI over documents (primary)",
                           "micro": "corpus aggregate: total tokens / total words"},
           "results": results}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{'tokenizer':34} {'vocab':>7} {'f(raw)':>8} {'f(norm)':>8} {'f(own)':>8} {'c(own)':>8}")
    for name, r in results.items():
        print(f"{name:34} {str(r['vocab_size']):>7} "
              f"{r['raw']['macro']['fertility']['mean']:8.4f} "
              f"{r['normalized']['macro']['fertility']['mean']:8.4f} "
              f"{r['own_norm']['macro']['fertility']['mean']:8.4f} "
              f"{r['own_norm']['macro']['compression']['mean']:8.4f}")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
