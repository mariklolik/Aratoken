import csv
import json

import numpy as np
import pytest

from scripts.lep.make_grid import COLUMNS
from scripts.lep.rebuttal_grid import rebuttal_rows


def manifest(tmp_path):
    path = tmp_path / "manifest.tsv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rebuttal_rows([1, 2, 3]))
    return path


def complete(root, row, values=(1., 2., 3.), text_hash="same"):
    run = root / "runs" / row["name"]
    (run / "model").mkdir(parents=True, exist_ok=True)
    (run / "model" / "run_metrics.json").write_text(json.dumps({
        "global_step": row["max_steps"], "checkpoint_selection": "final_step", "seed": row["seed"]}))
    for language in ("ar", "en"):
        documents = [{"index": index, "source_sha256": f"source{index}",
                      "text_sha256": f"{text_hash}{index}", "chars": 512, "bpc": value}
                     for index, value in enumerate(values)]
        data = {"n_docs": len(documents), "results": {row["name"]: {
            "bpc": {"mean": float(np.mean(values))}, "documents": documents,
            "protocol": {"name": "common-text-bpc-v1", "common_text_chars": 512,
                         "prefix_token_id": 1, "normalizer": "paper_normalizer(preserve_alif=True)"}}}}
        (run / f"bpc_{language}.json").write_text(json.dumps(data))


def test_empty_campaign_retains_all_78_runs_and_fixed_families(tmp_path):
    from scripts.eval.rebuttal_summary import summarize, markdown

    result = summarize(manifest(tmp_path), n_bootstrap=19)
    assert result["expected_runs"] == 78
    assert result["counts"] == {"complete": 0, "pending": 78, "failed": 0}
    assert len(result["runs"]) == 78
    assert len(result["groups"]) == 26
    assert len(result["families"]["fair"]["contrasts"]) == 6
    assert len(result["families"]["initializers"]["contrasts"]) == 12
    assert all(f["status"] == "incomplete" for f in result["families"].values())
    assert "budget500m_random" in markdown(result)


def test_failure_and_missing_seed_prevent_inference(tmp_path):
    from scripts.eval.rebuttal_summary import summarize

    path = manifest(tmp_path)
    for row in rebuttal_rows([1, 2, 3]):
        if row["name"] in {"small_hybrid_s1", "small_hybrid_s2", "small_cpt_s1", "small_cpt_s2"}:
            complete(tmp_path, row)
    failed = tmp_path / "runs" / "small_hybrid_s3"
    failed.mkdir()
    (failed / "failed.json").write_text(json.dumps({"error": "training failed"}))
    result = summarize(path, n_bootstrap=19, expected_docs={"ar": 3, "en": 3})
    assert result["counts"] == {"complete": 4, "pending": 73, "failed": 1}
    contrast = result["families"]["fair"]["contrasts"]["small_top4"]
    assert contrast["status"] == "incomplete"
    assert "inference" not in contrast
    assert result["groups"]["small_hybrid"]["ar"]["n_seeds"] == 2


def test_manifest_cannot_shrink_registered_denominator(tmp_path):
    from scripts.eval.rebuttal_summary import summarize

    path = manifest(tmp_path)
    path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError, match="78|manifest"):
        summarize(path, n_bootstrap=19)


def test_pairing_requires_same_canonical_documents_across_models_and_seeds(tmp_path):
    from scripts.eval.rebuttal_summary import summarize

    path = manifest(tmp_path)
    for row in rebuttal_rows([1, 2, 3]):
        if row["name"].startswith(("small_hybrid_s", "small_cpt_s")):
            complete(tmp_path, row, text_hash="different" if row["name"] == "small_cpt_s3" else "same")
    result = summarize(path, n_bootstrap=19, expected_docs={"ar": 3, "en": 3})
    contrast = result["families"]["fair"]["contrasts"]["small_top4"]
    assert contrast["status"] == "invalid_pairing"
    assert "inference" not in contrast
    assert "hash" in contrast["reason"]


def test_complete_family_reports_corrected_contrasts_without_dropping_arms(tmp_path):
    from scripts.eval.rebuttal_summary import summarize, markdown

    path = manifest(tmp_path)
    for row in rebuttal_rows([1, 2, 3]):
        offset = -0.5 if "hybrid" in row["name"] or "lep_all" in row["name"] else 0
        complete(tmp_path, row, values=tuple(v + offset for v in (1., 2., 3.)))
    result = summarize(path, n_bootstrap=19, expected_docs={"ar": 3, "en": 3})
    assert result["counts"]["complete"] == 78
    for family in result["families"].values():
        assert family["status"] == "complete"
        for contrast in family["contrasts"].values():
            inference = contrast["inference"]
            assert inference["mean_diff"] == pytest.approx(-.5)
            assert inference["p_raw"] == pytest.approx(.05)
            assert inference["p_holm"] >= inference["p_raw"]
            assert contrast["seed_deltas"] == {"1": -.5, "2": -.5, "3": -.5}
    assert "small_alpha025" in markdown(result)
    assert "scale17_calib" in markdown(result)


def test_crossed_bootstrap_keeps_seed_variability_and_is_reproducible():
    from scripts.eval.rebuttal_summary import crossed_bootstrap

    differences = np.array([[-1., -1.], [0., 0.], [1., 1.]])
    first = crossed_bootstrap(differences, n_bootstrap=199, seed=42)
    assert first == crossed_bootstrap(differences, n_bootstrap=199, seed=42)
    assert first["ci_lower"] < 0 < first["ci_upper"]
    assert first["p_value"] == 1


def test_complete_pair_cannot_claim_significance_before_family_finishes(tmp_path):
    from scripts.eval.rebuttal_summary import summarize

    path = manifest(tmp_path)
    for row in rebuttal_rows([1, 2, 3]):
        if row["name"].startswith(("small_hybrid_s", "small_cpt_s")):
            complete(tmp_path, row)
    result = summarize(path, n_bootstrap=19, expected_docs={"ar": 3, "en": 3})
    contrast = result["families"]["fair"]["contrasts"]["small_top4"]
    assert contrast["status"] == "complete"
    assert len(contrast["seed_deltas"]) == 3
    assert "inference" not in contrast


def test_expected_document_count_prevents_partial_evaluation_completion(tmp_path):
    from scripts.eval.rebuttal_summary import summarize

    path = manifest(tmp_path)
    complete(tmp_path, rebuttal_rows([1, 2, 3])[0])
    result = summarize(path, n_bootstrap=19)
    assert result["counts"] == {"complete": 0, "pending": 77, "failed": 1}
    assert "document count" in next(iter(result["runs"].values()))["failure"]["error"]
