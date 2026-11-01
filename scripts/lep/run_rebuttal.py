import argparse
from contextlib import contextmanager
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time

from scripts.eval.rebuttal_summary import read_evaluation


@contextmanager
def shared_lock(path: Path, wait: bool = False):
    while True:
        try:
            path.mkdir()
            break
        except FileExistsError:
            if not wait:
                yield False
                return
            time.sleep(1)
    owner = path / "owner.json"
    try:
        owner.write_text(json.dumps(dict(hostname=socket.gethostname(), pid=os.getpid(), timestamp=time.time())))
        yield True
    finally:
        owner.unlink(missing_ok=True)
        path.rmdir()


def source_digest(root: Path) -> str:
    paths = sorted((root / "aratoken").glob("*.py"))
    paths += [root / "scripts/lep/train.py", root / "scripts/eval/bpc.py"]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def bind_run(run_dir: Path, row: dict, manifest_hash: str, source_hash: str):
    path = run_dir / "run.json"
    expected = dict(row=row, manifest_sha256=manifest_hash, source_sha256=source_hash)
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise ValueError("Existing run provenance differs; use a separate run directory")
    else:
        if any((run_dir / name).exists() for name in ("model", "bpc_ar.json", "bpc_en.json", "complete.json")):
            raise ValueError("Existing artifacts have no matching run provenance")
        with path.open("x") as handle:
            json.dump(expected, handle, indent=2)


def initialized(model_dir: Path, spec: dict) -> bool:
    marker = model_dir / "initialization.json"
    if not marker.exists():
        return False
    if json.loads(marker.read_text()) != spec:
        raise ValueError("Cached initializer specification differs from manifest")
    return True


def evaluation_receipts(run_dir: Path, row: dict) -> dict:
    receipts = {}
    for language, count in (("ar", 1500), ("en", 1000)):
        path = run_dir / f"bpc_{language}.json"
        summary, evaluation = read_evaluation(path, row["name"], count)
        result = json.loads(path.read_text())["results"][row["name"]]
        protocol = evaluation["protocol"]
        required = {"character_unit": "Unicode code points", "normalizer": "paper_normalizer(preserve_alif=True), fixed point before character prefix",
                    "truncation": "canonical character prefix; no token truncation",
                    "scored_tokens": "all target tokens; prefix excluded"}
        if any(protocol.get(k) != value for k, value in required.items()):
            raise ValueError("Evaluation protocol is incomplete or differs from manifest")
        if (not isinstance(protocol["prefix_token_id"], int) or protocol["prefix_token_id"] < 0
                or not protocol["prefix_token"] or protocol["context_limit"] != 1024):
            raise ValueError("Evaluation conditioning or context limit is invalid")
        if (Path(result["model_path"]).resolve() != (run_dir / "model/final_model").resolve()
                or result["tokenizer_path"] != row["tokenizer"]):
            raise ValueError("Evaluation model or tokenizer differs from manifest")
        for document in evaluation["documents"]:
            if any(len(document[k]) != 64 or len(bytes.fromhex(document[k])) != 32
                   for k in ("source_sha256", "text_sha256")):
                raise ValueError("Evaluation document hash is invalid")
            if not (0 < document["chars"] <= 512 and 0 < document["tokens"] < 1024):
                raise ValueError("Evaluation character or token denominator is invalid")
            if (not math.isfinite(document["nll"]) or document["nll"] < 0
                    or not math.isclose(document["bpc"], document["nll"] / document["chars"] / math.log(2), rel_tol=1e-10)):
                raise ValueError("Evaluation NLL and BPC disagree")
        for unit in ("chars", "tokens"):
            if result[f"total_{unit}"] != sum(d[unit] for d in evaluation["documents"]):
                raise ValueError("Evaluation totals disagree with documents")
        if not math.isclose(result["bpc"]["mean"], summary["mean"], rel_tol=1e-10):
            raise ValueError("Evaluation mean differs from per-document values")
        receipts[language] = summary["sha256"]
    return receipts


def validate_completion(run_dir: Path, row: dict, manifest_hash: str, source_hash: str):
    marker = json.loads((run_dir / "complete.json").read_text())
    if (marker.get("manifest_sha256") != manifest_hash or marker.get("source_sha256") != source_hash
            or not completed_training(run_dir, int(row["max_steps"]))):
        raise ValueError("Cached completion provenance or training contract differs")
    if marker.get("evaluations") != evaluation_receipts(run_dir, row):
        raise ValueError("Cached completion evaluation digests differ")


def training_command(row: dict, run_dir: Path, python: str) -> list[str]:
    frozen = "false" if row["mode"] == "cpt" else "true"
    command = [python, "-m", "scripts.lep.train", f"seed={row['seed']}",
               f"model_path={row['model']}", f"tokenizer_path={row['tokenizer']}",
               f"data_path={row['data']}", f"freeze_old_embeddings={frozen}",
               f"include_lm_head_freeze={frozen}", f"unfreeze_transformer_layers={row['layers']}",
               f"training.learning_rate={row['lr']}", f"training.max_steps={row['max_steps']}",
               f"hydra.run.dir={run_dir}", *shlex.split(row["overrides"])]
    if list((run_dir / "model").glob("checkpoint-*/trainer_state.json")):
        command.append("resume=true")
    return command


def completed_training(run_dir: Path, steps: int) -> bool:
    metrics = run_dir / "model" / "run_metrics.json"
    if not metrics.exists() or not (run_dir / "model" / "final_model" / "config.json").exists():
        return False
    data = json.loads(metrics.read_text())
    return data.get("global_step") == steps and data.get("checkpoint_selection") == "final_step"


def record(path: Path, **data):
    data.update(timestamp=time.time(), hostname=socket.gethostname(), pid=os.getpid(),
                gpu=os.environ.get("CUDA_VISIBLE_DEVICES"))
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(json.dumps(data) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def execute(command: list[str], log: Path, events: Path, name: str, stage: str):
    record(events, run=name, stage=stage, state="started", command=command)
    with log.open("a") as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
    record(events, run=name, stage=stage, state="completed" if result.returncode == 0 else "failed",
           returncode=result.returncode, log=str(log))
    if result.returncode:
        raise RuntimeError(f"{name} {stage} failed; see {log}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("outputs/rebuttal/manifest.tsv"))
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--only", default="")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    root = args.manifest.parent.resolve()
    events = root / "events.jsonl"
    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    source_hash = source_digest(Path(__file__).resolve().parents[2])
    with args.manifest.open() as handle:
        matrix = list(csv.DictReader(handle, delimiter="\t"))
    specs = json.loads(args.manifest.with_suffix(".initializers.json").read_text())
    query = subprocess.check_output(["nvidia-smi", "-i", str(args.gpu),
                                     "--query-gpu=uuid,memory.used,utilization.gpu",
                                     "--format=csv,noheader,nounits"], text=True).strip().split(",")
    if int(query[1]) > 100 or int(query[2]) > 0:
        raise RuntimeError("selected GPU is occupied")
    lease_dir = Path.home() / ".gpu-leases"
    lease_dir.mkdir(exist_ok=True)
    with (lease_dir / f"{socket.gethostname()}-{query[0].strip()}.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.environ.update(CUDA_VISIBLE_DEVICES=query[0].strip(), OMP_NUM_THREADS="4",
                          MKL_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false",
                          HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        record(events, state="worker_started", manifest_sha256=manifest_hash)
        for row in matrix:
            name = row["name"]
            if args.only and args.only not in name:
                continue
            run_dir = root / "runs" / name
            run_dir.mkdir(parents=True, exist_ok=True)
            with shared_lock(run_dir / "worker.claim") as owned:
                if not owned:
                    continue
                bind_run(run_dir, row, manifest_hash, source_hash)
                if (run_dir / "complete.json").exists():
                    validate_completion(run_dir, row, manifest_hash, source_hash)
                    continue
                if (run_dir / "failed.json").exists() and not args.retry_failed:
                    continue
                try:
                    if row["model"] in specs:
                        spec = specs[row["model"]]
                        model_dir = Path(row["model"])
                        model_dir.parent.mkdir(parents=True, exist_ok=True)
                        with shared_lock(Path(str(model_dir) + ".claim"), wait=True):
                            marker = model_dir / "initialization.json"
                            if not initialized(model_dir, spec):
                                command = [sys.executable, "-m", "scripts.lep.init_embeddings"]
                                for key, value in spec.items():
                                    command.extend([f"--{key}", str(value)])
                                execute(command, Path(str(model_dir) + ".log"), events, name, "initialize")
                                marker.write_text(json.dumps(spec, indent=2))
                    if not completed_training(run_dir, int(row["max_steps"])):
                        execute(training_command(row, run_dir, sys.executable), run_dir / "train.log",
                                events, name, "train")
                    if not completed_training(run_dir, int(row["max_steps"])):
                        raise RuntimeError("training artifact is incomplete or uses the wrong checkpoint policy")
                    model = run_dir / "model" / "final_model"
                    for language, limit in (("ar", 1500), ("en", 1000)):
                        output = run_dir / f"bpc_{language}.json"
                        if output.exists():
                            continue
                        command = [sys.executable, "-m", "scripts.eval.bpc", "--model",
                                   f"{name}={model}:{row['tokenizer']}", "--lang", language,
                                   "--limit", str(limit), "--common-text-chars", "512",
                                   "--out", str(output)]
                        execute(command, run_dir / f"eval_{language}.log", events, name, f"evaluate_{language}")
                    receipts = evaluation_receipts(run_dir, row)
                    (run_dir / "complete.json").write_text(json.dumps(dict(timestamp=time.time(),
                        manifest_sha256=manifest_hash, source_sha256=source_hash,
                        evaluations=receipts, model=str(model))))
                    record(events, run=name, state="complete")
                except Exception as error:
                    (run_dir / "failed.json").write_text(json.dumps(dict(timestamp=time.time(), error=str(error))))
                    record(events, run=name, state="failed", error=str(error))
                    raise
        record(events, state="worker_finished", manifest_sha256=manifest_hash)


if __name__ == "__main__":
    main()
