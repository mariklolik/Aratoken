import argparse
import sys
from pathlib import Path

from aratoken.normalizer import paper_normalizer
from aratoken.paths import ARATOKEN, BASELINES

ROOT = Path(__file__).resolve().parents[2]

WORDS = [
    ("الاقتصاد", "definite article + derived noun"),
    ("أصعب", "hamzated comparative (Alif handling)"),
    ("مكتباتهم", "plural + possessive clitic"),
    ("وسيكتبونها", "conj + future + verb + object clitic"),
    ("التكنولوجيا", "long borrowing"),
]

SYSTEMS = [
    ("Ours (Alif pres.)", "sp", ARATOKEN["AraToken-150k-alif-preserve"][0], True),
    ("Ours (Alif unif.)", "sp", ARATOKEN["AraToken-150k-alif-unified"][0], False),
    ("AraBERT", "hf", BASELINES["AraBERT"], None),
    ("Qwen2.5/Qwen3", "hf", BASELINES["Qwen3"], None),
]


def load(kind: str, path: str):
    if kind == "sp":
        import sentencepiece as spm
        p = ROOT / path
        if not p.exists():
            return None
        sp = spm.SentencePieceProcessor(); sp.load(str(p))
        return lambda t: [x.replace("▁", "") or "_" for x in sp.encode(t, out_type=str)]
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    except Exception as exc:                                   # noqa: BLE001
        print(f"  ! {path}: {exc}", file=sys.stderr)
        return None
    def enc(t):
        out = []
        for piece in tok.tokenize(t):
            try:
                out.append(tok.convert_tokens_to_string([piece]).strip() or piece)
            except Exception:                                   # noqa: BLE001
                out.append(piece)
        return out
    return enc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path,
                   default=Path("paper/figures/segmentation_examples.pdf"))
    return p.parse_args()


def main() -> int:
    a = parse_args()
    encoders = []
    for name, kind, path, preserve in SYSTEMS:
        e = load(kind, path)
        if e is None:
            print(f"  skip {name}")
            continue
        norm = paper_normalizer(preserve) if preserve is not None else paper_normalizer(True)
        encoders.append((name, e, norm))

    rows = []
    print(f"{'word':16}", " | ".join(f"{n[:18]:18}" for n, _, _ in encoders))
    for w, gloss in WORDS:
        cells = []
        for name, enc, norm in encoders:
            pieces = enc(norm.normalize_str(w))
            cells.append(pieces)
        print(f"{w:16}", " | ".join(f"{'+'.join(c):18}" for c in cells))
        rows.append((w, gloss, cells))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import arabic_reshaper
    from bidi.algorithm import get_display

    def ar(t: str) -> str:
        return get_display(arabic_reshaper.reshape(t))

    ncol = len(encoders)
    nrow = len(rows)
    fig, ax = plt.subplots(figsize=(9.4, 0.60 * nrow + 0.80))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    T = ax.transAxes

    xs = [0.02] + [0.165 + i * (0.835 / ncol) for i in range(ncol)]
    row_h = 1.0 / (nrow + 1.15)
    y = 1.0 - row_h * 0.55

    ax.text(xs[0], y, "Word", fontsize=8.5, fontweight="bold", va="center", transform=T)
    for i, (name, _, _) in enumerate(encoders):
        ax.text(xs[i + 1], y, name, fontsize=8.5, fontweight="bold",
                va="center", transform=T)
    ax.plot([0, 1], [y - row_h * 0.42] * 2, color="#333", lw=0.9, transform=T,
            clip_on=False)

    for w, gloss, cells in rows:
        y -= row_h
        ax.text(xs[0], y + row_h * 0.20, ar(w), fontsize=10.5, va="center",
                transform=T)
        ax.text(xs[0], y - row_h * 0.32, gloss, fontsize=6.2, style="italic",
                color="#666", va="center", transform=T)
        for i, c in enumerate(cells):
            ax.text(xs[i + 1], y, ar(" + ".join(c)), fontsize=8.5,
                    va="center", transform=T)
        ax.plot([0, 1], [y - row_h * 0.5] * 2, color="#e0e0e0", lw=0.5,
                transform=T, clip_on=False)
    outp = ROOT / a.out if not a.out.is_absolute() else a.out
    outp = outp.with_suffix(".pdf")
    outp.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outp, bbox_inches="tight", dpi=300)
    plt.savefig(outp.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close()
    print(f"\nwrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
