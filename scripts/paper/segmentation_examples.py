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
    p.add_argument("--diagrams-outdir", type=Path)
    return p.parse_args()


def diagrams(outdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Rectangle
    from packaging.version import Version
    import arabic_reshaper
    from bidi.algorithm import get_display

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42})
    outdir.mkdir(parents=True, exist_ok=True)
    ink, blue, gray = "#203f68", "#dce8f4", "#ecf0f1"

    def canvas(height):
        figure = plt.figure(figsize=(3.45, height))
        axis = figure.add_axes([0, 0, 1, 1], xlim=(0, 1), ylim=(0, 1))
        axis.axis("off")
        return figure, axis

    def box(axis, x, y, width, height, title, body="", fill=blue, edge="#3a6fa5"):
        axis.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.006,rounding_size=0.013",
                                      facecolor=fill, edgecolor=edge, linewidth=1))
        axis.text(x + width / 2, y + height * (.72 if body else .5), title,
                  ha="center", va="center", color=ink, fontsize=8.2, weight="bold")
        if body:
            axis.text(x + width / 2, y + height * .29, body, ha="center", va="center",
                      color=ink, fontsize=7.6, linespacing=1.1)

    def arrow(axis, start, end):
        axis.annotate("", xy=end, xytext=start,
                      arrowprops={"arrowstyle": "->", "color": ink, "lw": 1})

    def save(figure, name):
        figure.savefig(outdir / f"{name}.pdf", bbox_inches="tight", pad_inches=.025)
        figure.savefig(outdir / f"{name}.png", bbox_inches="tight", pad_inches=.025, dpi=220)
        plt.close(figure)
        print(f"wrote {outdir / name}.pdf")

    fig, ax = canvas(4.0)
    for y, height, label in ((.425, .39, "Tokenizer training"), (.205, .195, "Model adaptation")):
        ax.add_patch(Rectangle((.02, y), .70, height, fill=False, edgecolor="#9aa7b5", linewidth=.9, linestyle="--"))
        ax.text(.045, y + height - .026, label, ha="left", va="center", fontsize=7.6, color="#627488")
    box(ax, .05, .865, .64, .105, "Arabic corpus", "FineWeb2-HQ Arabic", "#e4f0e8", "#4e9072")
    box(ax, .05, .595, .64, .175, "Normalization", "NFKC + rules; remove diacritics\nPreserve Alif (default)\nAlif unification optional")
    box(ax, .05, .445, .64, .105, "SentencePiece", "Unigram; 150K vocabulary")
    box(ax, .05, .225, .64, .125, "LEP", "Extend vocabulary; initialize rows\nTrain new rows + top layers")
    box(ax, .05, .045, .64, .105, "Arabic Qwen3", "Adapted model", "#fae9d7", "#e78025")
    box(ax, .765, .245, .205, .09, "Qwen3\nbase", fill=gray, edge="#9aa7b5")
    for start, end in ((.865, .770), (.595, .550), (.445, .350), (.225, .150)):
        arrow(ax, (.37, start), (.37, end))
    arrow(ax, (.765, .29), (.70, .29))
    save(fig, "overview_pipeline")

    fig, ax = canvas(4.6)
    box(ax, .035, .885, .44, .085, "Qwen3 vocabulary", "Original tokens", gray, "#9aa7b5")
    box(ax, .525, .885, .44, .085, "AraToken vocabulary", "New Arabic tokens")
    box(ax, .17, .765, .66, .08, "Extended vocabulary", "Original + added rows", "#fae9d7", "#e78025")
    arrow(ax, (.255, .885), (.40, .845))
    arrow(ax, (.745, .885), (.60, .845))
    box(ax, .10, .655, .80, .08, "Hybrid initialization", "Other initializers are controls")
    arrow(ax, (.50, .765), (.50, .735))
    ax.text(.50, .622, "Input embeddings", ha="center", va="center", fontsize=8.2, weight="bold", color=ink)
    box(ax, .06, .51, .53, .085, "Original rows", "Frozen", gray, "#9aa7b5")
    box(ax, .61, .51, .33, .085, "New rows", "Trainable")
    arrow(ax, (.50, .655), (.50, .635))
    box(ax, .12, .385, .76, .085, "Transformer layers 0-23", "Frozen", gray, "#9aa7b5")
    box(ax, .12, .265, .76, .085, "Transformer layers 24-27", "Trainable")
    arrow(ax, (.50, .51), (.50, .47))
    arrow(ax, (.50, .385), (.50, .35))
    ax.text(.50, .227, "LM head: tied to input embeddings", ha="center", va="center", fontsize=8, color=ink)
    box(ax, .06, .105, .53, .085, "Original rows", "Frozen", gray, "#9aa7b5")
    box(ax, .61, .105, .33, .085, "New rows", "Trainable")
    ax.text(.50, .045, "Same shared rows and gradient mask", ha="center", va="center", fontsize=7.6, color=ink)
    save(fig, "lep_architecture")

    fig, ax = canvas(2.65)
    reshaper = arabic_reshaper.ArabicReshaper(configuration={"delete_harakat": False, "delete_tatweel": False})
    native_rtl = Version(matplotlib.__version__) >= Version("3.11")
    examples = [("أَكَلَ", "اكل", "Optional Alif unification\n+ diacritics removed"),
                ("إِسْلَام", "اسلام", "Optional Alif unification\n+ diacritics removed"),
                ("آلاف", "الاف", "Optional Alif unification"),
                ("٠١٢٣", "0123", "Arabic-Indic to\nWestern numerals"),
                ("الـكـتـاب", "الكتاب", "Tatweel removed")]
    for x, title in ((.145, "Original"), (.40, "Normalized"), (.76, "Transformation")):
        ax.text(x, .95, title, ha="center", va="center", fontsize=8, weight="bold", color=ink)
    ax.plot([.02, .98], [.90, .90], color=ink, lw=.8)
    for index, (original, normalized, rule) in enumerate(examples):
        y = .80 - index * .17
        if index % 2 == 0:
            ax.add_patch(Rectangle((.02, y - .073), .96, .146, color=gray, ec="none"))
        for x, text, color in ((.145, original, ink), (.40, normalized, "#3a6fa5")):
            display = text if native_rtl else get_display(reshaper.reshape(text))
            ax.text(x, y, display, ha="center", va="center", fontsize=11, color=color)
        ax.text(.76, y, rule, ha="center", va="center", fontsize=7.6, linespacing=1.15, color=ink)
    save(fig, "normalization_examples")


def main() -> int:
    a = parse_args()
    if a.diagrams_outdir is not None:
        diagrams(ROOT / a.diagrams_outdir)
        return 0
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
