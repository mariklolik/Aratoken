import pytest

from scripts.lep.make_grid import COLUMNS, rows


def test_rebuttal_matrix_is_complete_and_separate():
    from scripts.lep.rebuttal_grid import rebuttal_rows

    matrix = rebuttal_rows([1, 2, 3])
    assert len(matrix) == 78
    assert len({r["name"] for r in matrix}) == len(matrix)
    assert all(set(r) == set(COLUMNS) for r in matrix)
    assert len(rows([1, 2, 3])) == 56
    for arm in ("small", "budget500m", "scale17"):
        for method in ("random", "mean", "calib", "analogical", "hybrid", "cpt", "lep_all", "cpt_all"):
            subset = [r for r in matrix if r["name"].startswith(f"{arm}_{method}_s")]
            assert {r["seed"] for r in subset} == {1, 2, 3}
            assert {r["max_steps"] for r in subset} == {5087 if arm == "budget500m" else 2000}


def test_baselines_have_embedding_freedom_and_final_step_selection():
    from scripts.lep.rebuttal_grid import rebuttal_rows

    for row in rebuttal_rows([1]):
        assert "training.load_best_model_at_end=false" in row["overrides"]
        assert "training.per_device_train_batch_size=8" in row["overrides"]
        assert "training.gradient_accumulation_steps=12" in row["overrides"]
        if "cpt" in row["name"] or "lep_all" in row["name"]:
            assert "train_embeddings=true" in row["overrides"]
            assert row["mode"] == "cpt"
        if "_all_" in row["name"]:
            assert row["layers"] == str(list(range(28))).replace(" ", "")


def test_initializer_seed_paths_and_alpha_endpoints():
    from scripts.lep.rebuttal_grid import initializer_specs, rebuttal_rows

    matrix = rebuttal_rows([1, 2, 3])
    specs = initializer_specs(matrix)
    assert len(specs) == 36
    for row in matrix:
        if row["model"].startswith("outputs/rebuttal-initializers/"):
            spec = specs[row["model"]]
            assert spec["seed"] == row["seed"]
            if "analogical" in row["name"]:
                assert spec["method"] == "hybrid" and spec["alpha"] == 0
            if "alpha025" in row["name"]:
                assert spec["alpha"] == .25
            if "alpha075" in row["name"]:
                assert spec["alpha"] == .75


@pytest.mark.parametrize("seeds", [[], [1, 1], [0], [-1]])
def test_invalid_seed_lists_fail(seeds):
    from scripts.lep.rebuttal_grid import rebuttal_rows

    with pytest.raises(ValueError):
        rebuttal_rows(seeds)


def test_worker_command_preserves_manifest_and_resumes_only_own_run(tmp_path):
    from scripts.lep.rebuttal_grid import rebuttal_rows
    from scripts.lep.run_rebuttal import training_command

    row = next(r for r in rebuttal_rows([2]) if r["name"] == "small_cpt_s2")
    command = training_command(row, tmp_path, "python")
    assert command[:3] == ["python", "-m", "scripts.lep.train"]
    assert "freeze_old_embeddings=false" in command
    assert "train_embeddings=true" in command
    assert "seed=2" in command
    assert "training.max_steps=2000" in command
    assert "resume=true" not in command
    (tmp_path / "model" / "checkpoint-1000").mkdir(parents=True)
    (tmp_path / "model" / "checkpoint-1000" / "trainer_state.json").write_text("{}")
    assert "resume=true" in training_command(row, tmp_path, "python")


def test_completion_rejects_missing_or_wrong_steps(tmp_path):
    import json
    from scripts.lep.run_rebuttal import completed_training

    assert not completed_training(tmp_path, 2000)
    (tmp_path / "model" / "final_model").mkdir(parents=True)
    (tmp_path / "model" / "final_model" / "config.json").write_text("{}")
    path = tmp_path / "model" / "run_metrics.json"
    path.write_text(json.dumps({"global_step": 1000}))
    assert not completed_training(tmp_path, 2000)
    path.write_text(json.dumps({"global_step": 2000, "checkpoint_selection": "final_step"}))
    assert completed_training(tmp_path, 2000)


def test_provenance_refuses_changed_manifest_row_or_source(tmp_path):
    from scripts.lep.run_rebuttal import bind_run

    row = {"name": "small_mean_s1", "seed": "1"}
    bind_run(tmp_path, row, "manifest", "source")
    original = (tmp_path / "run.json").read_bytes()
    bind_run(tmp_path, row, "manifest", "source")
    for changed_row, manifest, source in ((dict(row, seed="2"), "manifest", "source"),
                                           (row, "changed", "source"), (row, "manifest", "changed")):
        with pytest.raises(ValueError, match="provenance"):
            bind_run(tmp_path, changed_row, manifest, source)
        assert (tmp_path / "run.json").read_bytes() == original


def test_provenance_rejects_orphan_checkpoint(tmp_path):
    from scripts.lep.run_rebuttal import bind_run

    (tmp_path / "model").mkdir()
    with pytest.raises(ValueError, match="provenance"):
        bind_run(tmp_path, {"name": "run"}, "manifest", "source")


def test_initializer_marker_requires_identical_spec(tmp_path):
    import json
    from scripts.lep.run_rebuttal import initialized

    spec = {"method": "hybrid", "alpha": .5, "seed": 1}
    assert not initialized(tmp_path, spec)
    (tmp_path / "initialization.json").write_text(json.dumps(spec))
    assert initialized(tmp_path, spec)
    with pytest.raises(ValueError, match="initializer"):
        initialized(tmp_path, dict(spec, alpha=.25))


def test_scientific_source_digest_tracks_training_code_but_not_runner(tmp_path):
    from scripts.lep.run_rebuttal import source_digest

    for filename in ("aratoken/a.py", "scripts/lep/train.py", "scripts/eval/bpc.py", "scripts/lep/run_rebuttal.py"):
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x = 1\n")
    original = source_digest(tmp_path)
    (tmp_path / "scripts/lep/run_rebuttal.py").write_text("x = 2\n")
    assert source_digest(tmp_path) == original
    (tmp_path / "aratoken/a.py").write_text("x = 2\n")
    assert source_digest(tmp_path) != original


def test_completion_refuses_marker_without_evaluation_receipts(tmp_path):
    import json
    from scripts.lep.run_rebuttal import validate_completion

    (tmp_path / "complete.json").write_text(json.dumps({"manifest_sha256": "manifest"}))
    with pytest.raises(ValueError):
        validate_completion(tmp_path, {"name": "run", "max_steps": "2000"}, "manifest", "source")


def test_evaluation_rejects_legacy_or_short_artifact(tmp_path):
    import json
    from scripts.lep.run_rebuttal import evaluation_receipts

    (tmp_path / "bpc_ar.json").write_text(json.dumps({"n_docs": 1, "results": {"run": {"bpc": {"mean": 1}}}}))
    with pytest.raises((ValueError, KeyError)):
        evaluation_receipts(tmp_path, {"name": "run", "tokenizer": "tok"})


def test_complete_receipt_validates_full_artifacts_and_detects_changed_digest(tmp_path):
    import hashlib
    import json
    import math
    from scripts.lep.run_rebuttal import evaluation_receipts, validate_completion

    row = {"name": "run", "tokenizer": "tok", "max_steps": "2000"}
    model = tmp_path / "model" / "final_model"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    (model.parent / "run_metrics.json").write_text(json.dumps({
        "global_step": 2000, "checkpoint_selection": "final_step"}))
    protocol = {"name": "common-text-bpc-v1", "common_text_chars": 512,
                "character_unit": "Unicode code points", "normalizer": "paper_normalizer(preserve_alif=True), fixed point before character prefix",
                "truncation": "canonical character prefix; no token truncation", "prefix_token_id": 1,
                "prefix_token": "[BOS]", "context_limit": 1024,
                "scored_tokens": "all target tokens; prefix excluded"}
    for language, count in (("ar", 1500), ("en", 1000)):
        documents = [{"index": i, "source_sha256": hashlib.sha256(str(i).encode()).hexdigest(),
                      "text_sha256": hashlib.sha256(b"a").hexdigest(), "chars": 1, "tokens": 1,
                      "nll": math.log(2), "bpc": 1.} for i in range(count)]
        result = {"documents": documents, "protocol": protocol, "bpc": {"mean": 1.},
                  "model_path": str(model), "tokenizer_path": "tok", "total_chars": count, "total_tokens": count}
        (tmp_path / f"bpc_{language}.json").write_text(json.dumps({"n_docs": count, "results": {"run": result}}))
    marker = {"manifest_sha256": "manifest", "source_sha256": "source",
              "evaluations": evaluation_receipts(tmp_path, row)}
    (tmp_path / "complete.json").write_text(json.dumps(marker))
    validate_completion(tmp_path, row, "manifest", "source")
    marker["evaluations"]["ar"] = "changed"
    (tmp_path / "complete.json").write_text(json.dumps(marker))
    with pytest.raises(ValueError, match="completion"):
        validate_completion(tmp_path, row, "manifest", "source")


def test_shared_lock_excludes_contenders_and_cleans_up(tmp_path):
    import json
    from scripts.lep.run_rebuttal import shared_lock

    claim = tmp_path / "claim"
    with shared_lock(claim) as owned:
        assert owned
        assert json.loads((claim / "owner.json").read_text())["pid"] > 0
        with shared_lock(claim) as contender:
            assert not contender
        assert claim.exists()
    assert not claim.exists()
    with shared_lock(claim) as owned:
        assert owned


def test_shared_lock_releases_on_exception(tmp_path):
    from scripts.lep.run_rebuttal import shared_lock

    claim = tmp_path / "claim"
    with pytest.raises(RuntimeError, match="interrupted"):
        with shared_lock(claim):
            raise RuntimeError("interrupted")
    assert not claim.exists()
