import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
import numpy as np                                             # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.labelsize": 10, "axes.titlesize": 10,
                     "figure.dpi": 300, "savefig.dpi": 300})
ALGO_COLOR = {"bpe": "#2ecc71", "sp": "#3498db", "wp": "#e74c3c"}
WIDTH = 3.4


PNG = False


def save(fig, outdir: Path, name: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight")
    if PNG:
        fig.savefig(outdir / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outdir / name}.pdf")


def load(name: str) -> dict | None:
    p = RESULTS / name
    if not p.exists():
        print(f"  skip {name}: not measured yet")
        return None
    return json.loads(p.read_text())


def algorithm_bars(outdir: Path) -> None:
    d = load("intrinsic_algorithms.json")
    if d is None:
        return
    rows = [("drop", "", "diacritics removed"),
            ("drop", "_norm", "diacritics removed\n+ character rules"),
            ("keep", "", "diacritics kept"),
            ("keep", "_norm", "diacritics kept\n+ character rules")]
    algos = [("bpe", "BPE", "#95a5a6", "s"), ("wp", "WordPiece", "#7f8c8d", "^"),
             ("sp", "Unigram (AraToken)", "#16a085", "o")]
    for metric, label, fname in (
            ("fertility", "Tokens per word (lower is better)",
             "fertility_comparison"),
            ("compression", "Characters per token (higher is better)",
             "compression_comparison")):
        fig, ax = plt.subplots(figsize=(WIDTH, 2.2))
        y = np.arange(len(rows))
        for a, lab, col, mk in algos:
            xs, lo, hi = [], [], []
            for dia, n, _ in rows:
                m = d["results"].get(f"{a}_{dia}{n}", {}).get("own_norm", {}) \
                    .get("macro", {}).get(metric)
                xs.append(m["mean"] if m else np.nan)
                lo.append(m["mean"] - m["ci_lower"] if m else 0)
                hi.append(m["ci_upper"] - m["mean"] if m else 0)
            ours = a == "sp"
            ax.errorbar(xs, y, xerr=[lo, hi], fmt=mk, ms=6.5 if ours else 5,
                        color=col, ecolor=col, elinewidth=0.8, capsize=2,
                        label=lab, zorder=3 if ours else 2)
        ax.set_yticks(y)
        ax.set_yticklabels([r[2] for r in rows], fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel(label, fontsize=8)
        ax.grid(axis="x", alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.legend(fontsize=6.5, ncol=3, loc="lower center",
                  bbox_to_anchor=(0.45, 1.0), frameon=False,
                  handletextpad=0.3, columnspacing=0.8)
        save(fig, outdir, fname)


def pruning_chart(outdir: Path) -> None:
    d = load("intrinsic_metrics.json")
    if d is None:
        return
    keys = [("AraToken-150k-alif-preserve", "full"),
            ("AraToken-99pct-alif-preserve", "99% cov."),
            ("AraToken-95pct-alif-preserve", "95% cov.")]
    keys = [(k, lab) for k, lab in keys if k in d["results"]]
    if not keys:
        return
    vocab = [d["results"][k]["vocab_size"] for k, _ in keys]
    fert = [d["results"][k]["own_norm"]["macro"]["fertility"]["mean"] for k, _ in keys]
    comp = [d["results"][k]["own_norm"]["macro"]["compression"]["mean"] for k, _ in keys]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.3))
    ax.plot(vocab, fert, "o-", color="#3498db", label="fertility")
    ax.set_xlabel("vocabulary size")
    ax.set_ylabel("fertility", color="#3498db")
    ax.tick_params(axis="y", labelcolor="#3498db")
    ax2 = ax.twinx()
    ax2.plot(vocab, comp, "s--", color="#e67e22", label="compression")
    ax2.set_ylabel("chars / token", color="#e67e22")
    ax2.tick_params(axis="y", labelcolor="#e67e22")
    for v, f, (_, lab) in zip(vocab, fert, keys):
        ax.annotate(f"{lab}\n{v:,}", (v, f), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    save(fig, outdir, "vocab_pruning")


def init_shrinkage(outdir: Path) -> None:
    d = load("init_norm_shrinkage.json")
    if d is None:
        return
    ks = sorted(int(k) for k in d["by_subtoken_count"])
    ratio = [d["by_subtoken_count"][str(k)]["ratio_to_donor"] for k in ks]
    pred = [k ** -0.5 for k in ks]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.3))
    ax.plot(ks, ratio, "o-", color="#c0392b", label="measured")
    ax.plot(ks, pred, "k--", linewidth=1, label=r"$|S_k|^{-1/2}$ (uncorrelated)")
    ax.axhline(1.0, color="#2c3e50", linewidth=1, alpha=0.6)
    ax.annotate("donor embeddings", (ks[-1], 1.0), textcoords="offset points",
                xytext=(-4, 4), ha="right", fontsize=7, color="#2c3e50")
    ax.set_xlabel("sub-tokens per new token, $|S_k|$")
    ax.set_ylabel("radius / donor radius")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    save(fig, outdir, "init_shrinkage")


def bpc_bars(outdir: Path) -> None:
    d = load("grid_bpc_with_init.json")
    if d is None:
        return
    agg = d["by_config"]
    rows = [("factorial_preserve_lr2e-4", "mean sub-token (Cui et al.)"),
            ("init_focus", "FOCUS-style"), ("init_wechsel", "WECHSEL-style"),
            ("init_hybrid", "hybrid (ours)")]
    rows = [(k, lab) for k, lab in rows if k in agg]
    if not rows:
        return
    vals = [agg[k]["mean"] for k, _ in rows]
    errs = [agg[k]["std"] or 0.0 for k, _ in rows]
    ours = [k == "init_hybrid" for k, _ in rows]
    fig, ax = plt.subplots(figsize=(WIDTH, 1.7))
    y = np.arange(len(rows))
    for yi, v, e, o in zip(y, vals, errs, ours):
        ax.errorbar(v, yi, xerr=e, fmt="o", ms=7 if o else 5, capsize=3,
                    color="#16a085" if o else "#7f8c8d", ecolor="#555",
                    elinewidth=0.8)
    best_base = min(v for v, o in zip(vals, ours) if not o)
    if any(ours):
        h = vals[ours.index(True)]
        ax.annotate(f"$-${best_base - h:.4f} BPC vs. best baseline",
                    xy=(h, y[ours.index(True)]), xytext=(6, -11),
                    textcoords="offset points", fontsize=6.5, ha="right",
                    color="#16a085", fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([lab for _, lab in rows], fontsize=7)
    for t, o in zip(ax.get_yticklabels(), ours):
        if o:
            t.set_fontweight("bold")
            t.set_color("#16a085")
    ax.set_xlabel("Arabic BPC at 2,000 steps (lower is better)", fontsize=8)
    lo, hi = min(vals) - 0.012, max(vals) + 0.004
    ax.set_xlim(lo, hi)
    ax.invert_xaxis()
    ax.grid(axis="x", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    save(fig, outdir, "init_bpc")


def training_curves(outdir: Path) -> None:
    src = RESULTS / "curves"
    runs = [("flagship_preserve.json", "LEP (extended vocabulary)", "#16a085"),
            ("flagship_cpt.json", "CPT (original tokenizer)", "#7f8c8d")]
    have = [(f, lab, c) for f, lab, c in runs if (src / f).exists()]
    if not have:
        print("  skip training_curves: no trainer_state")
        return
    fig, axes = plt.subplots(1, len(have), figsize=(WIDTH, 2.2), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (fname, label, colour) in zip(axes, have):
        h = json.loads((src / fname).read_text())["log_history"]
        tr = [(e["step"], e["loss"]) for e in h if "loss" in e]
        ev = [(e["step"], e["eval_loss"]) for e in h if "eval_loss" in e]
        if tr:
            ax.plot(*zip(*tr), color=colour, linewidth=1, label="train")
        if ev:
            ax.plot(*zip(*ev), color=colour, linewidth=1.2, linestyle="--",
                    marker="o", markersize=2, label="dev")
        ax.set_title(label, fontsize=7)
        ax.set_xlabel("step")
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("cross-entropy / token")
    axes[0].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    save(fig, outdir, "training_curves")


def adaptation_bars(outdir: Path) -> None:
    d = load("grid_bpc_with_init.json")
    if d is None:
        return
    agg = d["by_config"]
    rows = [("trainable_L27", "LEP, layer 27"),
            ("trainable_L24_27", "LEP, layers 24-27"),
            ("trainable_L18_27", "LEP, layers 18-27"),
            ("trainable_L9_27", "LEP, layers 9-27"),
            ("trainable_all", "LEP, all layers")]
    rows = [(k, lab) for k, lab in rows if k in agg]
    cpt = agg.get("baseline_cpt_origtok", {}).get("mean")
    if not rows or cpt is None:
        return
    vals = [agg[k]["mean"] for k, _ in rows]
    errs = [agg[k]["std"] or 0.0 for k, _ in rows]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.2))
    y = np.arange(len(rows))
    ax.barh(y, [cpt - v for v in vals], left=vals, height=0.62,
            color="#16a085", edgecolor="white", linewidth=0.5)
    ax.errorbar(vals, y, xerr=errs, fmt="none", ecolor="#333", elinewidth=0.8,
                capsize=2)
    ax.axvline(cpt, color="#555", linestyle="--", linewidth=1)
    ax.text(cpt, -0.72, "CPT, original tokenizer", ha="right", va="bottom",
            fontsize=6.5, color="#555", clip_on=False)
    for yi, v in zip(y, vals):
        ax.text(v - 0.004, yi, f"$-${cpt - v:.2f}", ha="right", va="center",
                fontsize=7, color="#16a085", fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([lab for _, lab in rows], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Arabic BPC, 2,000 steps (lower is better)", fontsize=8)
    ax.set_xlim(min(vals) - 0.035, cpt + 0.008)
    ax.grid(axis="x", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    save(fig, outdir, "ablation_study")


def baseline_bars(outdir: Path) -> None:
    d = load("intrinsic_metrics.json")
    if d is None:
        return
    r = d["results"]
    rows = [("AraToken-150k-alif-preserve", "AraToken\n(150K)"),
            ("AraToken-95pct-alif-preserve", "AraToken\n(60K)"),
            ("AraBERT", "AraBERT\n(64K)"),
            ("Qwen2.5", "Qwen2.5/3"),
            ("DeepSeek-V3", "DeepSeek")]
    rows = [(k, lab) for k, lab in rows if k in r]
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH * 2, 2.3))
    for ax, metric, label, better in (
            (axes[0], "fertility", "tokens / word", "lower is better"),
            (axes[1], "compression", "chars / token", "higher is better")):
        vals, los, his = [], [], []
        for k, _ in rows:
            m = r[k]["normalized"]["macro"][metric]
            vals.append(m["mean"])
            los.append(m["mean"] - m["ci_lower"])
            his.append(m["ci_upper"] - m["mean"])
        best = (min if metric == "fertility" else max)(range(len(vals)),
                                                      key=vals.__getitem__)
        ax.bar(np.arange(len(rows)), vals, yerr=[los, his], capsize=3,
               edgecolor="white", linewidth=0.5,
               color=["#16a085" if i == best else "#95a5a6"
                      for i in range(len(rows))])
        ax.set_xticks(np.arange(len(rows)))
        ax.set_xticklabels([lab for _, lab in rows], fontsize=7,
                           rotation=30, ha="right")
        ax.set_ylabel(label)
        ax.set_title(better, fontsize=8, color="#555")
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, outdir, "baseline_comparison")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=Path,
                   default=Path("paper/figures"))
    p.add_argument("--png", action="store_true",
                   help="Also write PNGs, for looking at a figure without a "
                        "PDF viewer. They are not part of the manuscript.")
    return p.parse_args()


def main() -> int:
    global PNG
    a = parse_args()
    PNG = a.png
    outdir = a.outdir if a.outdir.is_absolute() else ROOT / a.outdir
    algorithm_bars(outdir)
    baseline_bars(outdir)
    pruning_chart(outdir)
    init_shrinkage(outdir)
    bpc_bars(outdir)
    training_curves(outdir)
    adaptation_bars(outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
