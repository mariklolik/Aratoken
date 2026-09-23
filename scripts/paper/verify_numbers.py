import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"
PAPER = ROOT / "paper"
tex = PAPER.joinpath("manuscript.tex").read_text()
for t in sorted(PAPER.joinpath("tables").glob("*.tex")):
    tex += t.read_text()

def j(name: str):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None

checks = []
def chk(label, expected, fmt="{:.4f}"):
    if expected is None:
        checks.append((label, "SOURCE MISSING", "", False)); return
    want = fmt.format(expected).lstrip("-")
    found = want in tex
    checks.append((label, want, "in paper" if found else "NOT FOUND", found))

g = j("grid_bpc_with_init.json")
if g:
    a = g["by_config"]
    for k, lab in (("__base_qwen3_0.6b", "base BPC"), ("baseline_cpt_origtok", "CPT BPC"),
                   ("factorial_preserve_lr2e-4", "LEP L24-27 BPC"),
                   ("trainable_all", "all-layers BPC"), ("baseline_embonly", "emb-only BPC"),
                   ("init_wechsel", "WECHSEL BPC"), ("init_focus", "FOCUS BPC"),
                   ("init_random", "random BPC")):
        chk(lab, a.get(k, {}).get("mean"))
s17 = j("grid_bpc_scale17.json")
if s17:
    a = s17["by_config"]
    for k, lab in (("__base_qwen3_1.7b", "1.7B base BPC"), ("scale17full_cpt", "1.7B CPT BPC"),
                   ("scale17hyb_lep", "1.7B LEP BPC")):
        chk(lab, a.get(k, {}).get("mean"))
    chk("1.7B LEP gain vs base", a["scale17hyb_lep"]["mean"] - a["__base_qwen3_1.7b"]["mean"], "{:.3f}")
    chk("1.7B CPT gain vs base", a["scale17full_cpt"]["mean"] - a["__base_qwen3_1.7b"]["mean"], "{:.3f}")
st = j("stats_scale17.json")
if st:
    c = st["pairwise_contrasts"]["scale17hyb_lep_minus_scale17full_cpt"]
    checks.append(("1.7B LEP-CPT p<0.001", f"{c['p_value']}", "", c["p_value"] < 0.001 and c["ci_upper"] < 0))
f = j("bpc_flagship.json")
if f:
    for k, lab in (("flagship_preserve_500M", "flagship LEP BPC"), ("flagship_cpt_500M", "flagship CPT BPC")):
        chk(lab, f["results"].get(k, {}).get("bpc", {}).get("mean"))
e = j("bpc_english_flagship.json")
if e:
    for k, lab in (("flagship_cpt_500M", "flagship CPT En BPC"), ("flagship_preserve_500M", "flagship LEP En BPC")):
        chk(lab, e["results"].get(k, {}).get("bpc", {}).get("mean"))
m = j("minhash_leakage.json")
if m: chk("minhash rate %", 100 * m["near_duplicate_rate"], "{:.2f}")
i = j("intrinsic_metrics.json")
if i:
    r = i["results"]
    def f_norm(k): return r.get(k, {}).get("normalized", {}).get("macro", {}).get("fertility", {}).get("mean")
    def f_own(k): return r.get(k, {}).get("own_norm", {}).get("macro", {}).get("fertility", {}).get("mean")
    chk("AraToken pres fertility", f_norm("AraToken-150k-alif-preserve"), "{:.3f}")
    chk("Qwen2.5 fertility", f_norm("Qwen2.5"), "{:.3f}")
    chk("AraBERT fertility", f_norm("AraBERT"), "{:.3f}")
    q, a150 = f_norm("Qwen2.5"), f_norm("AraToken-150k-alif-preserve-bytes") or f_norm("AraToken-150k-alif-preserve")
    ab, a60 = f_own("AraBERT"), f_own("AraToken-95pct-alif-preserve")
    if ab and a60: chk("reduction vs AraBERT at 60K %", 100*(1-a60/ab), "{:.1f}")
alg = j("intrinsic_algorithms.json")
oa = j("stats_oall.json")
ds = j("downstream.json")
gi = j("grid_bpc_with_init.json")
st = j("stats.json")
if alg:
    r = alg["results"]
    for k in ("sp_drop_norm", "bpe_drop_norm", "wp_drop_norm"):
        chk(f"{k} fertility", r.get(k, {}).get("own_norm", {}).get("macro", {}).get("fertility", {}).get("mean"), "{:.3f}")
d = j("dialectal_fertility.json")
if d and "AraToken-150k-alif-preserve-bytes" in d:
    chk("dialect fertility (bytes)", d["AraToken-150k-alif-preserve-bytes"]["dialectal_arsarcasm"]["fertility_mean"], "{:.2f}")
n = j("init_norm_shrinkage.json")
if n: chk("mean-init radius ratio", n["mean_init_ratio_to_donor"], "{:.2f}")

i = j("intrinsic_metrics.json")
alg = j("intrinsic_algorithms.json")
dia = j("dialectal_fertility.json")
if i:
    R = i["results"]
    fn = lambda k: R[k]["normalized"]["macro"]["fertility"]["mean"]
    fo = lambda k: R[k]["own_norm"]["macro"]["fertility"]["mean"]
    full, p99, p95 = ("AraToken-150k-alif-preserve",
                      "AraToken-99pct-alif-preserve",
                      "AraToken-95pct-alif-preserve")
    chk("reduction vs Qwen2.5 %", 100 * (1 - fn(full) / fn("Qwen2.5")), "{:.1f}")
    chk("reduction vs AraBERT, 150K %", 100 * (1 - fo(full) / fo("AraBERT")), "{:.1f}")
    chk("reduction vs AraBERT, 60K %", 100 * (1 - fo(p95) / fo("AraBERT")), "{:.1f}")
    chk("pruning 99%: vocab removed %", 100 * (1 - R[p99]["vocab_size"] / 150000), "{:.0f}")
    chk("pruning 99%: fertility cost %", 100 * (fo(p99) / fo(full) - 1), "{:.1f}")
    chk("pruning 95%: fertility cost %", 100 * (fo(p95) / fo(full) - 1), "{:.1f}")
if alg:
    ao = lambda k: alg["results"][k]["own_norm"]["macro"]["fertility"]["mean"]
    for a in ("bpe", "wp", "sp"):
        chk(f"{a}: whole pipeline %", 100 * (ao(f"{a}_drop_norm") / ao(f"{a}_keep") - 1), "{:.1f}")
        chk(f"{a}: replacements %", 100 * (ao(f"{a}_drop_norm") / ao(f"{a}_drop") - 1), "{:.2f}")
if dia and "AraToken-150k-alif-preserve-bytes" in dia:
    d = lambda k: dia[k]["dialectal_arsarcasm"]["fertility_mean"]
    chk("dialectal advantage %", 100 * (1 - d("AraToken-150k-alif-preserve-bytes") / d("Qwen3")), "{:.1f}")

if gi:
    bc = gi["by_config"]
    for k, lab in (("init_hybrid", "hybrid"), ("init_wechsel", "WECHSEL"),
                   ("init_focus", "FOCUS"), ("init_calib", "moment-matched"),
                   ("factorial_preserve_lr2e-4", "mean sub-token"),
                   ("init_random", "random")):
        if k in bc:
            chk(f"init BPC {lab}", bc[k]["mean"], "{:.4f}")
if st:
    for k, lab in (("init_hybrid_minus_init_wechsel", "hybrid-WECHSEL"),
                   ("init_hybrid_minus_init_focus", "hybrid-FOCUS"),
                   ("init_hybrid_minus_factorial_preserve_lr2e-4", "hybrid-mean")):
        c = (st.get("pairwise_contrasts") or {}).get(k)
        if c:
            chk(f"init delta {lab}", abs(c["mean_diff"]), "{:.4f}")

if oa:
    for tag in ("base", "cpt", "lep"):
        chk(f"OALL pooled {tag}", oa["accuracy"][tag]["pooled"], "{:.4f}")
        chk(f"OALL macro {tag}", oa["accuracy"][tag]["macro"], "{:.3f}")
    c = oa["pooled"]["cpt:lep"]
    chk("OALL cpt-lep delta pts", 100 * abs(c["delta_accuracy"]), "{:.1f}")
    chk("OALL cpt-lep p", c["p_value"], "{:.2f}")

if ds:
    g = lambda a, t, sh, k: ds[a][t][sh][k]
    chk("downstream cpt sarcasm 0shot",
        g("cpt_flagship_500m", "ArSarcasm-Sarcasm", "0shot", "macro_f1")["mean"], "{:.3f}")
    chk("downstream lep sanad 3shot",
        g("lep_flagship_500m", "sanad", "3shot", "macro_f1")["mean"], "{:.3f}")
    chk("downstream floor sarcasm 0shot",
        g("cpt_flagship_500m", "ArSarcasm-Sarcasm", "0shot", "chance")["label_prior"], "{:.3f}")
    chk("downstream floor sanad 3shot",
        g("lep_flagship_500m", "sanad", "3shot", "chance")["label_prior"], "{:.3f}")

bad = [c for c in checks if not c[3]]
for lab, want, where, ok in checks:
    print(f"{'ok ' if ok else 'MISS'} {lab:34} {want:>10} {where}")
print(f"\n{len(checks) - len(bad)}/{len(checks)} headline numbers match")
sys.exit(1 if bad else 0)
