import csv
import importlib.util
import json
from pathlib import Path

import pytest

from scripts.lep.rebuttal_grid import rebuttal_rows


def exporter(name="export"):
    path = Path(__file__).resolve().parents[1] / "scripts/paper/rebuttal_tables.py"
    assert path.exists(), "Complete-campaign table exporter is missing"
    spec = importlib.util.spec_from_file_location("rebuttal_tables", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, name)


def summary():
    result = {"expected_runs": 78, "counts": {"complete": 78, "pending": 0, "failed": 0},
              "expected_docs": {"ar": 1500, "en": 1000}, "runs": {}, "groups": {},
              "families": {}, "manifest_sha256": "test-manifest"}
    for row in rebuttal_rows([1, 2, 3]):
        group = row["name"].rsplit("_s", 1)[0]
        result["runs"][row["name"]] = {
            "name": row["name"], "seed": row["seed"], "status": "complete",
            "evaluations": {language: {"mean": value, "n_docs": count}
                            for language, value, count in (("ar", 1.25, 1500), ("en", 1.5, 1000))},
            "metrics": {"train_runtime": 12.25, "peak_allocated_bytes": 123456,
                        "trainable_layers": [24, 25, 26, 27], "seed": row["seed"]}}
        result["groups"][group] = {
            language: {"n_seeds": 3, "values": {"1": value, "2": value, "3": value},
                       "mean": value, "std": 0.0}
            for language, value in (("ar", 1.25), ("en", 1.5))}
    for family, count in (("fair", 6), ("initializers", 12)):
        contrasts = {}
        for setting in ("small", "budget500m", "scale17"):
            comparisons = (("top4", "hybrid", "cpt"), ("all", "lep_all", "cpt_all")) if family == "fair" else (
                (f"hybrid_minus_{method}", "hybrid", method)
                for method in ("random", "mean", "calib", "analogical"))
            for name, left, right in comparisons:
                contrasts[f"{setting}_{name}"] = {
                    "left": f"{setting}_{left}", "right": f"{setting}_{right}", "status": "complete",
                    "inference": {"mean_diff": -.0123, "ci_lower": -.02, "ci_upper": .001,
                                  "p_holm": .6, "n_seeds": 3, "n_documents": 1500}}
        result["families"][family] = {"status": "complete", "planned_contrasts": count,
                                     "contrasts": contrasts}
    return result


@pytest.mark.parametrize("incomplete", ["runs", "fair", "initializers"])
def test_incomplete_campaign_cannot_write_publishable_tables(tmp_path, incomplete):
    result = summary()
    if incomplete == "runs":
        result["counts"] = {"complete": 77, "pending": 1, "failed": 0}
    else:
        result["families"][incomplete]["status"] = "incomplete"
    destination = tmp_path / "tables"
    with pytest.raises(ValueError, match="complete|78"):
        exporter()(result, destination)
    assert not destination.exists()


def test_export_preserves_all_runs_metrics_zero_sd_and_full_contrasts(tmp_path):
    exporter()(summary(), tmp_path)
    with (tmp_path / "per-seed.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 78
    assert {row["name"] for row in rows} == {row["name"] for row in rebuttal_rows([1, 2, 3])}
    assert rows[0]["arabic_bpc"] == "1.25"
    assert rows[0]["english_bpc"] == "1.5"
    assert rows[0]["metric.train_runtime"] == "12.25"
    assert json.loads(rows[0]["metric.trainable_layers"]) == [24, 25, 26, 27]
    assert "1.2500 $\\pm$ 0.0000" in (tmp_path / "fair-estimates.tex").read_text()
    for name, count in (("fair-inference.tex", 6), ("initializer-inference.tex", 12)):
        table = (tmp_path / name).read_text()
        assert table.count("-0.0123 & [-0.0200, +0.0010] & 0.6") == count
        assert "marginal" in table and "Holm" in table
    alpha = (tmp_path / "alpha-sweep.tex").read_text()
    assert "0.00 & Character-ngram analogical proxy" in alpha
    assert "1.00 & Compositional mean" in alpha
    assert alpha.split(r"\caption", 1)[0].count("$\\pm$") == 10


def test_incomplete_seed_group_is_rejected_before_writing(tmp_path):
    result = summary()
    result["groups"]["small_mean"]["ar"]["n_seeds"] = 2
    with pytest.raises(ValueError, match="seed"):
        exporter()(result, tmp_path / "tables")


def downstream_summary():
    from scripts.eval.downstream_summary import MODELS, PAIRS, TASKS

    counts = {"ar": dict(zip(TASKS["ar"], (1000, 977, 537, 387, 1763))),
              "en": {task: 1000 for task in TASKS["en"]}}
    result = {"status": "complete", "planned_contrasts": 9, "expected_bpc_docs": 1000,
              "cohort_counts": counts, "models": {}, "contrasts": {}}
    for tag, run in MODELS.items():
        seed = int(tag[-1]) if run else None
        accuracy = 50.0 + (seed or 0)
        result["models"][tag] = {
            "run": run, "training_seed": seed,
            "tasks": {language: {task: {"status": "complete", "accuracy": accuracy,
                                       "n": counts[language][task], "sha256": "task-digest"}
                                 for task in tasks} for language, tasks in TASKS.items()},
            "macro_accuracy": {"ar": accuracy, "en": accuracy},
            "en_bpc": {"status": "complete", "mean": 1.2 + (seed or 0) / 100, "n_docs": 1000}}
    for endpoint in ("ar_accuracy", "en_accuracy", "en_bpc"):
        for pair, (left, right) in PAIRS.items():
            result["contrasts"][f"{endpoint}:{pair}"] = {
                "status": "complete", "endpoint": endpoint, "left": left, "right": right,
                "units": "bits per character" if endpoint == "en_bpc" else "percentage points",
                "inference": {"mean_diff": .01 if endpoint == "en_bpc" else 2.0,
                              "ci_lower": -.1, "ci_upper": 3., "p_holm": .6,
                              "n_seeds": 3, "ci_scope": "marginal"}}
    return result


@pytest.mark.parametrize("incomplete", ["family", "models", "task", "bpc"])
def test_downstream_requires_full_nine_contrast_family_and_49_blocks(tmp_path, incomplete):
    result = downstream_summary()
    if incomplete == "family":
        result["contrasts"]["en_bpc:lep-base"].pop("inference")
    elif incomplete == "models":
        result["models"].pop("l500t3")
    elif incomplete == "task":
        result["models"]["l500t3"]["tasks"]["ar"]["ArabicMMLU"]["status"] = "pending"
    else:
        result["models"]["b06"]["en_bpc"]["status"] = "pending"
    with pytest.raises(ValueError, match="[Cc]omplete|seven|49"):
        exporter("export_downstream")(result, tmp_path / "tables")
    assert not (tmp_path / "tables").exists()


def test_downstream_keeps_fixed_base_49_rows_and_native_accuracy_units(tmp_path):
    exporter("export_downstream")(downstream_summary(), tmp_path)
    with (tmp_path / "downstream-per-checkpoint-task.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 49
    base = [row for row in rows if row["checkpoint"] == "b06"]
    assert len(base) == 7
    assert all(row["training_seed"] == "" for row in base)
    assert all(row["accuracy"] == "50.0" for row in base)
    estimates = (tmp_path / "downstream-estimates.tex").read_text()
    assert "ArabicMMLU & 1,000 & 50.0000 & 52.0000 $\\pm$ 1.0000 & 52.0000 $\\pm$ 1.0000" in estimates
    assert "Arabic macro" in estimates and "4,664" in estimates
    inference = (tmp_path / "downstream-inference.tex").read_text()
    assert inference.count("+2.0000 & [-0.1000, +3.0000]") == 6
    assert inference.count("+0.0100 & [-0.1000, +3.0000]") == 3
    assert "percentage points" in inference and "fixed tasks" in inference
