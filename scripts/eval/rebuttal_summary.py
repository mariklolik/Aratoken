import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from aratoken.stats import holm
from scripts.lep.rebuttal_grid import rebuttal_rows


def crossed_bootstrap(differences: np.ndarray, n_bootstrap: int = 10000, seed: int = 42) -> dict:
    differences = np.asarray(differences, dtype=float)
    if differences.ndim != 2 or differences.shape[0] != 3 or differences.shape[1] < 1:
        raise ValueError("Three paired seeds and nonempty shared documents are required")
    if n_bootstrap < 1 or not np.isfinite(differences).all():
        raise ValueError("Finite differences and positive bootstrap count are required")
    rng = np.random.default_rng(seed)
    boot = np.empty(n_bootstrap)
    for start in range(0, n_bootstrap, 128):
        count = min(128, n_bootstrap - start)
        seeds = rng.integers(0, 3, size=(count, 3))
        docs = rng.integers(0, differences.shape[1], size=(count, differences.shape[1]))
        boot[start:start + count] = differences[seeds[:, :, None], docs[:, None, :]].mean(axis=(1, 2))
    observed = float(differences.mean())
    exceedances = np.count_nonzero(np.abs(boot - boot.mean()) >= abs(observed))
    return {"mean_diff": observed, "ci_lower": float(np.percentile(boot, 2.5)),
            "ci_upper": float(np.percentile(boot, 97.5)),
            "p_value": float((exceedances + 1) / (n_bootstrap + 1)),
            "n_seeds": 3, "n_documents": differences.shape[1], "n_bootstrap": n_bootstrap,
            "rng_seed": seed, "resampling": "crossed paired seeds and documents"}


def seed_summary(values: dict) -> dict:
    array = np.asarray(list(values.values()), dtype=float)
    return {"n_seeds": len(values), "values": values,
            "mean": float(array.mean()) if len(array) else None,
            "std": float(array.std(ddof=1)) if len(array) > 1 else None}


def read_evaluation(path: Path, name: str, expected_docs: int) -> tuple[dict, dict]:
    data = json.loads(path.read_text())
    result = data["results"][name]
    documents, protocol = result["documents"], result["protocol"]
    if protocol["name"] != "common-text-bpc-v1" or protocol["common_text_chars"] != 512:
        raise ValueError("Evaluation does not use the registered 512-character protocol")
    if len(documents) != data["n_docs"] or len(documents) != expected_docs:
        raise ValueError("Evaluation document count is incomplete")
    values = np.asarray([d["bpc"] for d in documents], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Evaluation contains nonfinite BPC")
    for index, document in enumerate(documents):
        if document["index"] != index or not document["text_sha256"] or not document["source_sha256"]:
            raise ValueError("Evaluation document indices or hashes are missing")
    summary = {"mean": float(values.mean()), "n_docs": len(documents), "protocol": protocol,
               "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return summary, {"values": values, "documents": documents, "protocol": protocol}


def load_run(root: Path, row: dict, expected_docs: dict) -> tuple[dict, dict]:
    directory = root / "runs" / row["name"]
    report = {"name": row["name"], "seed": int(row["seed"]), "status": "pending",
              "missing": [], "evaluations": {}, "metrics": {}}
    cached = {}
    try:
        metrics = directory / "model" / "run_metrics.json"
        if metrics.exists():
            report["metrics"] = json.loads(metrics.read_text())
            if (report["metrics"].get("global_step") != int(row["max_steps"])
                    or report["metrics"].get("checkpoint_selection") != "final_step"
                    or report["metrics"].get("seed") != int(row["seed"])):
                raise ValueError("Training step, checkpoint policy, or seed differs from manifest")
        else:
            report["missing"].append("run_metrics.json")
        for language in ("ar", "en"):
            path = directory / f"bpc_{language}.json"
            if path.exists():
                report["evaluations"][language], cached[language] = read_evaluation(
                    path, row["name"], expected_docs[language])
            else:
                report["missing"].append(path.name)
        if not report["missing"]:
            report["status"] = "complete"
        elif (directory / "failed.json").exists():
            report.update(status="failed", failure=json.loads((directory / "failed.json").read_text()))
    except (ValueError, KeyError, TypeError) as error:
        report.update(status="failed", failure={"error": str(error)})
    return report, cached


def paired_differences(left: str, right: str, cached: dict) -> np.ndarray:
    reference = None
    differences = []
    for seed in (1, 2, 3):
        values = []
        for group in (left, right):
            evaluation = cached[f"{group}_s{seed}"]["ar"]
            signature = [(d["source_sha256"], d["text_sha256"], d["chars"])
                         for d in evaluation["documents"]]
            identity = (signature, evaluation["protocol"])
            if reference is None:
                reference = identity
            elif identity != reference:
                raise ValueError("Canonical document hashes, character counts, or protocols differ")
            values.append(evaluation["values"])
        differences.append(values[0] - values[1])
    return np.asarray(differences)


def summarize(manifest: Path, n_bootstrap: int = 10000, expected_docs: dict | None = None) -> dict:
    expected_docs = {"ar": 1500, "en": 1000} if expected_docs is None else expected_docs
    if set(expected_docs) != {"ar", "en"} or any(n < 1 for n in expected_docs.values()):
        raise ValueError("Positive expected document counts are required for ar and en")
    with manifest.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    expected = [{k: str(v) for k, v in row.items()} for row in rebuttal_rows([1, 2, 3])]
    if len(rows) != 78 or sorted(rows, key=lambda r: r["name"]) != sorted(expected, key=lambda r: r["name"]):
        raise ValueError("Manifest must contain all 78 registered runs with unchanged settings")
    runs, cached, groups = {}, {}, {}
    for row in rows:
        name = row["name"]
        runs[name], cached[name] = load_run(manifest.parent, row, expected_docs)
        groups.setdefault(name.rsplit("_s", 1)[0], []).append(name)
    group_reports = {}
    for group, names in groups.items():
        report = {"counts": {status: sum(runs[n]["status"] == status for n in names)
                              for status in ("complete", "pending", "failed")}}
        for language in ("ar", "en"):
            report[language] = seed_summary({str(runs[n]["seed"]): runs[n]["evaluations"][language]["mean"]
                                            for n in names if language in runs[n]["evaluations"]})
        report["training_runtime_seconds"] = sum(runs[n]["metrics"].get("train_runtime", 0) for n in names)
        report["max_peak_allocated_bytes"] = max(runs[n]["metrics"].get("peak_allocated_bytes", 0)
                                                 for n in names)
        group_reports[group] = report
    definitions = {"fair": {}, "initializers": {}}
    for setting in ("small", "budget500m", "scale17"):
        definitions["fair"][f"{setting}_top4"] = (f"{setting}_hybrid", f"{setting}_cpt")
        definitions["fair"][f"{setting}_all"] = (f"{setting}_lep_all", f"{setting}_cpt_all")
        for method in ("random", "mean", "calib", "analogical"):
            definitions["initializers"][f"{setting}_hybrid_minus_{method}"] = (
                f"{setting}_hybrid", f"{setting}_{method}")
    families = {}
    for family, definitions_by_name in definitions.items():
        contrasts, paired = {}, {}
        for name, (left, right) in definitions_by_name.items():
            incomplete = [f"{group}_s{s}" for group in (left, right) for s in (1, 2, 3)
                          if runs[f"{group}_s{s}"]["status"] != "complete"]
            contrast = {"left": left, "right": right, "status": "incomplete", "incomplete_runs": incomplete}
            if not incomplete:
                try:
                    paired[name] = paired_differences(left, right, cached)
                    delta = seed_summary({str(s + 1): float(v.mean()) for s, v in enumerate(paired[name])})
                    contrast.update(status="complete", seed_deltas=delta["values"],
                                    seed_delta_mean=delta["mean"], seed_delta_std=delta["std"])
                except ValueError as error:
                    contrast.update(status="invalid_pairing", reason=str(error))
            contrasts[name] = contrast
        complete = all(c["status"] == "complete" for c in contrasts.values())
        if complete:
            statistics = {name: crossed_bootstrap(values, n_bootstrap) for name, values in paired.items()}
            corrected = holm({name: values["p_value"] for name, values in statistics.items()})
            for name in contrasts:
                contrasts[name]["inference"] = {**statistics[name], **corrected[name]}
        families[family] = {"status": "complete" if complete else "incomplete", "metric": "Arabic BPC",
                            "planned_contrasts": len(contrasts), "contrasts": contrasts}
    return {"manifest": str(manifest), "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "expected_runs": 78, "expected_docs": expected_docs,
            "counts": {s: sum(r["status"] == s for r in runs.values())
                                             for s in ("complete", "pending", "failed")},
            "uncertainty": "Descriptive seed SD; inference only for complete three-seed families",
            "runs": runs, "groups": group_reports, "families": families}


def markdown(result: dict) -> str:
    counts = result["counts"]
    lines = [f"Runs: {counts['complete']}/78 complete; {counts['pending']} pending; {counts['failed']} failed.",
             "", "Arabic and English BPC are seed means ± seed SD. Missing seeds remain explicit.", "",
             "| Arm | Complete/pending/failed | Arabic BPC (seeds) | English BPC (seeds) |",
             "|---|---|---|---|"]
    for name, group in result["groups"].items():
        cells = []
        for language in ("ar", "en"):
            item = group[language]
            deviation = f" ± {item['std']:.4f}" if item["std"] is not None else ""
            cells.append(f"{item['mean']:.4f}{deviation} ({item['n_seeds']}/3)" if item["mean"] is not None else "— (0/3)")
        status = "/".join(str(group["counts"][s]) for s in ("complete", "pending", "failed"))
        lines.append(f"| {name} | {status} | {cells[0]} | {cells[1]} |")
    lines += ["", "Primary Arabic contrasts; negative deltas favor the first arm.", "",
              "| Family / contrast | Status | Delta [95% CI] | Holm p |", "|---|---|---|---|"]
    for name, family in result["families"].items():
        for key, contrast in family["contrasts"].items():
            inference = contrast.get("inference")
            estimate = (f"{inference['mean_diff']:.4f} [{inference['ci_lower']:.4f}, {inference['ci_upper']:.4f}]"
                        if inference else "—")
            pvalue = f"{inference['p_holm']:.6g}" if inference else "—"
            status = contrast["status"] if inference else f"{contrast['status']}; family {family['status']}"
            lines.append(f"| {name} / {key} | {status} | {estimate} | {pvalue} |")
    lines += ["", "Incomplete families have no inferential claims. English is descriptive; nonsignificance is not equivalence."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("outputs/rebuttal/manifest.tsv"))
    parser.add_argument("--out", type=Path, default=Path("outputs/rebuttal/summary.json"))
    parser.add_argument("--expected-ar-docs", type=int, default=1500)
    parser.add_argument("--expected-en-docs", type=int, default=1000)
    args = parser.parse_args()
    result = summarize(args.manifest, expected_docs={"ar": args.expected_ar_docs, "en": args.expected_en_docs})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    args.out.with_suffix(".md").write_text(markdown(result))
    print(json.dumps(result["counts"]))


if __name__ == "__main__":
    main()
