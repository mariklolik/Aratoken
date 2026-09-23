import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import sentencepiece as spm
from transformers.convert_slow_tokenizer import import_protobuf

from aratoken.normalizer import build_qwen_like_normalizer, config_from_spec
from aratoken.splits import load_texts

SPECIAL = {"<unk>", "<s>", "</s>", "<pad>", "<mask>"}


def piece_frequencies(tokenizer_dir: Path, texts: list[str]) -> Counter:
    spec = json.loads((tokenizer_dir / "tokenizer_config.json").read_text())["normalizer_spec"]
    normalizer = build_qwen_like_normalizer(config_from_spec(spec))
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_dir / "sentencepiece.model"))
    freq = Counter({sp.id_to_piece(i): 0 for i in range(sp.get_piece_size())})
    for text in texts:
        for piece in sp.encode(normalizer.normalize_str(text), out_type=str):
            if sp.piece_to_id(piece) != sp.unk_id() and piece in freq:
                freq[piece] += 1
    return freq


def covering_pieces(ranked: list[tuple[str, int]], coverage: float) -> set[str]:
    total = sum(f for _, f in ranked)
    keep, cum = set(), 0
    for piece, f in ranked:
        keep.add(piece)
        cum += f
        if cum / total >= coverage:
            break
    return keep


def prune(tokenizer_dir: Path, keep: set[str], coverage: float) -> Path:
    model = import_protobuf().ModelProto()
    model.ParseFromString((tokenizer_dir / "sentencepiece.model").read_bytes())
    keep |= {p.piece for p in model.pieces
             if p.piece in SPECIAL or (p.piece.startswith("<|") and p.piece.endswith("|>"))}
    kept = [p for p in model.pieces if p.piece in keep]
    del model.pieces[:]
    model.pieces.extend(kept)
    out = tokenizer_dir.parent / f"{tokenizer_dir.name}_{coverage:.2f}".replace(".", "_")
    out.mkdir(exist_ok=True)
    (out / "sentencepiece.model").write_bytes(model.SerializeToString())
    shutil.copy2(tokenizer_dir / "tokenizer_config.json", out / "tokenizer_config.json")
    print(f"{out}: kept {len(kept)} pieces for {coverage:.0%} coverage")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("tokenizer_dir", type=Path)
    p.add_argument("--coverage", type=float, nargs="+", default=[0.99, 0.95])
    p.add_argument("--shard-dir", type=Path, default=Path("data/fineweb2hq-ar/tok-train"))
    a = p.parse_args()
    freq = piece_frequencies(a.tokenizer_dir, load_texts(a.shard_dir))
    ranked = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    with (a.tokenizer_dir / "freq.csv").open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows([("token", "frequency"), *ranked])
    for c in a.coverage:
        prune(a.tokenizer_dir, covering_pieces(ranked, c), c)


if __name__ == "__main__":
    main()
