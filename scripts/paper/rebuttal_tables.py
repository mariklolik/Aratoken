import argparse
import csv
import hashlib
import json
from pathlib import Path

from scripts.eval.downstream_summary import MODELS, PAIRS, TASKS
from scripts.eval.rebuttal_summary import seed_summary
from scripts.lep.rebuttal_grid import rebuttal_rows

SETTINGS = {"small": ("0.6B", "2,000"), "budget500m": ("0.6B", "5,087"),
            "scale17": ("1.7B", "2,000")}
METHODS = {"mean": "Compositional mean", "calib": "Moment-matched mean",
           "analogical": "Character-ngram analogical proxy", "hybrid": "Half mixture (ours)",
           "random": "Random"}
ARABIC = (r"Arabic macro document BPC on 1,500 shared canonical documents, each truncated "
          r"to at most 512 Unicode characters. Lower BPC is better. ")
SEEDS = r"Values are means $\pm$ sample standard deviation (SD) across three paired training seeds. "
HELDOUT = r"The held-out data had previously been inspected. "
INFERENCE = (r"Intervals are 95\% marginal crossed seed/document bootstrap intervals "
             r"over three paired training seeds and 1,500 shared documents; "
             r"they are not simultaneous intervals. ")


def validate(result: dict) -> None:
    rows = rebuttal_rows([1, 2, 3])
    expected = {row["name"] for row in rows}
    if (result["expected_runs"] != 78 or result["counts"] != {"complete": 78, "pending": 0, "failed": 0}
            or set(result["runs"]) != expected
            or any(run["status"] != "complete" for run in result["runs"].values())):
        raise ValueError("All 78 registered runs must be complete before table export")
    if result["expected_docs"] != {"ar": 1500, "en": 1000}:
        raise ValueError("Canonical Arabic/English document counts must be 1500/1000")
    groups = {name.rsplit("_s", 1)[0] for name in expected}
    if set(result["groups"]) != groups:
        raise ValueError("All registered seed groups are required")
    for group in result["groups"].values():
        for language in ("ar", "en"):
            item = group[language]
            if item["n_seeds"] != 3 or set(item["values"]) != {"1", "2", "3"} or item["std"] is None:
                raise ValueError("Each group requires three seeds and their sample SD")
    for family, count in (("fair", 6), ("initializers", 12)):
        item = result["families"][family]
        if item["status"] != "complete" or item["planned_contrasts"] != count or len(item["contrasts"]) != count:
            raise ValueError(f"The complete {count}-contrast {family} family is required")
        for contrast in item["contrasts"].values():
            inference = contrast.get("inference", {})
            if (contrast["status"] != "complete" or inference.get("n_seeds") != 3
                    or inference.get("n_documents") != 1500
                    or not {"mean_diff", "ci_lower", "ci_upper", "p_holm"} <= inference.keys()):
                raise ValueError("Complete paired inference is required for every contrast")


def mean_sd(group: dict, language: str) -> str:
    item = group[language]
    return rf"{item['mean']:.4f} $\pm$ {item['std']:.4f}"


def table(headers: list[str], rows: list[list[str]], caption: str, label: str) -> str:
    return "\n".join([
        r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}" + "l" * len(headers) + r"@{}}", r"\toprule",
        " & ".join(headers) + r" \\", r"\midrule",
        *(" & ".join(row) + r" \\" for row in rows), r"\bottomrule", r"\end{tabular}",
        r"\caption{" + caption + "}", r"\label{" + label + "}", r"\end{table*}", "",
    ])


def inference_cells(contrast: dict) -> list[str]:
    item = contrast["inference"]
    return [f"{item['mean_diff']:+.4f}", f"[{item['ci_lower']:+.4f}, {item['ci_upper']:+.4f}]",
            f"{item['p_holm']:.6g}"]


def export(result: dict, output: Path) -> list[str]:
    validate(result)
    groups, families = result["groups"], result["families"]
    estimates, fair_inference, initializer_inference = [], [], []
    for setting, (size, steps) in SETTINGS.items():
        for policy, label in (("top4", "Top four"), ("all", "Full")):
            contrast = families["fair"]["contrasts"][f"{setting}_{policy}"]
            estimates.append([size, steps, label, *[
                mean_sd(groups[contrast[arm]], language)
                for language in ("ar", "en") for arm in ("right", "left")]])
            fair_inference.append([size, steps, label, *inference_cells(contrast)])
        for method in ("mean", "calib", "analogical", "random"):
            contrast = families["initializers"]["contrasts"][f"{setting}_hybrid_minus_{method}"]
            initializer_inference.append([size, steps, METHODS[method], *inference_cells(contrast)])
    contents = {
        "fair-estimates.tex": table(
            ["Model", "Steps", "Trainability", r"CPT Arabic $\downarrow$", r"LEP Arabic $\downarrow$",
             r"CPT English $\downarrow$", r"LEP English $\downarrow$"],
            estimates, ARABIC + SEEDS + r"English uses 1,000 canonical documents with the same prefix rule "
            r"and is descriptive. Top-four LEP updates new embedding rows and the last four transformer layers; "
            r"CPT updates all original embedding rows and the same layers. Full pairs update all parameters. "
            r"LEP (ours) uses the predeclared half mixture. Matched nominal packed-token budgets do not imply equal compute "
            r"or text exposure. " + HELDOUT, "tab:rebuttal-fair-estimates"),
        "fair-inference.tex": table(
            ["Model", "Steps", "Trainability", r"$\Delta$ BPC $\downarrow$", r"95\% interval", r"Holm $p$"],
            fair_inference, ARABIC + r"LEP (ours) minus CPT for all six trainability comparisons. "
            r"Negative differences favor LEP. " + INFERENCE + r"Holm correction covers the six planned "
            r"fairness contrasts. " + HELDOUT, "tab:rebuttal-fair-inference"),
        "initializer-inference.tex": table(
            ["Model", "Steps", "Comparator", r"$\Delta$ BPC $\downarrow$", r"95\% interval", r"Holm $p$"],
            initializer_inference, ARABIC + r"Half-mixture LEP (ours) minus each initializer comparator, with "
            r"the top-four training policy. Negative differences favor the half mixture. " + INFERENCE
            + r"Holm correction covers all 12 planned initializer contrasts. " + HELDOUT,
            "tab:rebuttal-initializer-inference"),
    }
    for language, title in (("ar", "Arabic"), ("en", "English")):
        rows = [[label, *[mean_sd(groups[f"{setting}_{method}"], language) for setting in SETTINGS]]
                for method, label in METHODS.items()]
        scope = ARABIC if language == "ar" else (
            r"English macro document BPC on 1,000 shared canonical documents, each truncated to at most "
            r"512 Unicode characters. Lower BPC is better. English results are descriptive only. ")
        contents[f"initializer-{language}.tex"] = table(
            ["Initializer", r"\shortstack{0.6B / 2,000 steps\\BPC $\downarrow$}",
             r"\shortstack{0.6B / 5,087 steps\\BPC $\downarrow$}",
             r"\shortstack{1.7B / 2,000 steps\\BPC $\downarrow$}"],
            rows, scope + SEEDS + r"All five predeclared initializers use the top-four training policy. "
            r"The analogical initializer is a character-ngram proxy. No initializer is selected from "
            r"these results. " + HELDOUT, f"tab:rebuttal-initializer-{language}")
    sweep = [(0., "analogical", METHODS["analogical"]), (.25, "alpha025", "Quarter compositional"),
             (.5, "hybrid", METHODS["hybrid"]), (.75, "alpha075", "Three-quarters compositional"),
             (1., "mean", METHODS["mean"])]
    contents["alpha-sweep.tex"] = table(
        [r"$\alpha$", "Initializer", r"Arabic BPC $\downarrow$", r"English BPC $\downarrow$"],
        [[f"{alpha:.2f}", label, *[mean_sd(groups[f"small_{method}"], language) for language in ("ar", "en")]]
         for alpha, method, label in sweep],
        ARABIC + SEEDS + r"Complete exploratory five-point mixture sweep at 0.6B, 2,000 steps, top-four "
        r"training. $\alpha$ weights the compositional component; $1-\alpha$ weights the character-ngram "
        r"analogical proxy. Endpoints reuse the component runs. The predeclared primary value remains "
        r"$\alpha=0.5$. English uses 1,000 canonical documents with the same prefix rule and is descriptive. "
        + HELDOUT, "tab:rebuttal-alpha")
    metric_keys = sorted({key for run in result["runs"].values() for key in run["metrics"]})
    fields = ["name", "seed", "status", "arabic_bpc", "english_bpc", "arabic_n_docs", "english_n_docs"]
    fields += [f"metric.{key}" for key in metric_keys]
    output.mkdir(parents=True, exist_ok=True)
    for filename, content in contents.items():
        (output / filename).write_text(content)
    with (output / "per-seed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for name, run in sorted(result["runs"].items()):
            row = {key: run[key] for key in ("name", "seed", "status")}
            for language, prefix in (("ar", "arabic"), ("en", "english")):
                row[f"{prefix}_bpc"] = run["evaluations"][language]["mean"]
                row[f"{prefix}_n_docs"] = run["evaluations"][language]["n_docs"]
            for key, value in run["metrics"].items():
                row[f"metric.{key}"] = json.dumps(value) if isinstance(value, (dict, list)) else value
            writer.writerow(row)
    return [*contents, "per-seed.csv"]


def validate_downstream(result: dict) -> None:
    expected = {f"{endpoint}:{pair}" for endpoint in ("ar_accuracy", "en_accuracy", "en_bpc") for pair in PAIRS}
    if (result["status"] != "complete" or result["planned_contrasts"] != 9
            or set(result["contrasts"]) != expected or result["expected_bpc_docs"] != 1000):
        raise ValueError("The complete nine-contrast downstream family is required")
    if set(result["models"]) != set(MODELS):
        raise ValueError("All seven registered downstream models are required")
    for tag, model in result["models"].items():
        if model["run"] != MODELS[tag] or model["training_seed"] != (int(tag[-1]) if MODELS[tag] else None):
            raise ValueError("The seven model identities and training seeds must match registration")
        if model["en_bpc"]["status"] != "complete" or model["en_bpc"]["n_docs"] != 1000:
            raise ValueError("Complete English BPC is required for all seven models")
        for language, tasks in TASKS.items():
            entries = model["tasks"][language]
            if set(entries) != set(tasks) or any(
                entries[task]["status"] != "complete" or entries[task]["n"] != result["cohort_counts"][language][task]
                for task in tasks
            ):
                raise ValueError("All 49 downstream task blocks must be complete with identical counts")
    for key, contrast in result["contrasts"].items():
        inference = contrast.get("inference", {})
        units = "bits per character" if key.startswith("en_bpc:") else "percentage points"
        if (contrast["status"] != "complete" or contrast["units"] != units
                or inference.get("n_seeds") != 3 or inference.get("ci_scope") != "marginal"
                or not {"mean_diff", "ci_lower", "ci_upper", "p_holm"} <= inference.keys()):
            raise ValueError("Complete native-unit marginal inference is required for all nine contrasts")


def export_downstream(result: dict, output: Path) -> list[str]:
    validate_downstream(result)
    models, counts = result["models"], result["cohort_counts"]
    estimates, records, inference_rows = [], [], []

    def cells(values: dict) -> list[str]:
        return [f"{values['b06']:.4f}", *[
            mean_sd({"value": seed_summary({str(seed): values[f"{prefix}500t{seed}"] for seed in (1, 2, 3)})}, "value")
            for prefix in ("c", "l")]]

    for language, tasks in TASKS.items():
        for task in tasks:
            estimates.append([r"$\uparrow$ " + task, f"{counts[language][task]:,}", *cells({
                tag: model["tasks"][language][task]["accuracy"] for tag, model in models.items()})])
            for tag, model in models.items():
                records.append({"checkpoint": tag, "training_seed": model["training_seed"], "run": model["run"],
                                "language": language, "task": task, **model["tasks"][language][task]})
    for language, label in (("ar", "Arabic macro"), ("en", "English macro")):
        estimates.append([r"$\uparrow$ " + label, f"{sum(counts[language].values()):,}", *cells({
            tag: model["macro_accuracy"][language] for tag, model in models.items()})])
    estimates.append([r"$\downarrow$ English BPC", "1,000", *cells({tag: model["en_bpc"]["mean"] for tag, model in models.items()})])
    for endpoint, label in (("ar_accuracy", r"$\uparrow$ Arabic macro"), ("en_accuracy", r"$\uparrow$ English macro"),
                            ("en_bpc", r"$\downarrow$ English BPC")):
        for pair, (left, right) in PAIRS.items():
            contrast = result["contrasts"][f"{endpoint}:{pair}"]
            units = "BPC" if endpoint == "en_bpc" else "percentage points"
            inference_rows.append([label, pair.upper(), units, *inference_cells(contrast)])
    caption = (r"Qwen3-0.6B with LEP (ours): fixed base and three training seeds per adapted arm at 5,087 steps, top-four policy. "
               r"Base is one fixed checkpoint, not three independent runs. Adapted values are seed means $\pm$ "
               r"sample standard deviation (SD). Task accuracies and task macros are percentages (higher is better); macros weight "
               r"the five Arabic or two English tasks equally. $n$ is the exact item count per task, summed "
               r"item count for each macro, or document count for BPC. English BPC uses canonical prefixes "
               r"of at most 512 Unicode characters (lower is better). All 49 checkpoint/task blocks are included. "
               + HELDOUT)
    contents = {
        "downstream-estimates.tex": table(
            ["Endpoint", r"$n$", "Base (fixed)", "CPT", "LEP"], estimates, caption,
            "tab:rebuttal-downstream-estimates"),
        "downstream-inference.tex": table(
            ["Endpoint", "Contrast", "Units", "Difference", r"95\% interval", r"Holm $p$"],
            inference_rows, r"All nine predeclared downstream contrasts including LEP (ours); differences are left minus right. "
            r"Positive accuracy differences (percentage points) and negative BPC differences favor the left arm. "
            r"95\% marginal bootstrap intervals share three paired training-seed draws and resample paired "
            r"items within the seven fixed tasks, without task resampling; English BPC uses crossed paired "
            r"seed/document resampling on 1,000 documents. Base is held fixed in every draw. Holm correction "
            r"covers all nine contrasts; intervals are not simultaneous. Nonsignificance does not establish "
            r"equivalence. " + HELDOUT, "tab:rebuttal-downstream-inference"),
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, content in contents.items():
        (output / name).write_text(content)
    filename = "downstream-per-checkpoint-task.csv"
    with (output / filename).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for record in records for key in record}), lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, (dict, list)) else value
                         for key, value in record.items()} for record in records)
    return [*contents, filename]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=Path("outputs/rebuttal/summary.json"))
    parser.add_argument("--out", type=Path, default=Path("outputs/rebuttal/paper-tables"))
    parser.add_argument("--downstream", type=Path)
    args = parser.parse_args()
    source = args.summary.read_bytes()
    result = json.loads(source)
    downstream_source = args.downstream.read_bytes() if args.downstream else None
    try:
        if downstream_source is not None:
            downstream = json.loads(downstream_source)
            validate_downstream(downstream)
        filenames = export(result, args.out)
        if downstream_source is not None:
            filenames += export_downstream(downstream, args.out)
    except (KeyError, ValueError) as error:
        parser.exit(2, f"Tables not exported: {error}\n")
    receipt = {"summary": str(args.summary), "summary_sha256": hashlib.sha256(source).hexdigest(),
               "manifest_sha256": result["manifest_sha256"], "counts": result["counts"],
               "files": {name: hashlib.sha256((args.out / name).read_bytes()).hexdigest() for name in filenames}}
    if downstream_source is not None:
        receipt.update(downstream_summary=str(args.downstream),
                       downstream_summary_sha256=hashlib.sha256(downstream_source).hexdigest())
    (args.out / "provenance.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"output": str(args.out), "files": [*filenames, "provenance.json"]}))


if __name__ == "__main__":
    main()
