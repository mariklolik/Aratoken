import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoTokenizer

from aratoken.normalizer import canonical_text
from aratoken.splits import load_texts
from aratoken.training import load_causal_lm


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, metavar="TAG=MODEL[:TOKENIZER]")
    parser.add_argument("--split", type=Path, default=Path("data/fineweb2hq-ar/test"))
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--prompt-chars", type=int, default=256)
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.limit, args.prompt_chars, args.new_tokens, args.warmup, *args.batch_sizes) < 1:
        parser.error("counts, batch sizes, and warmup must be positive")
    if not args.device.startswith("cuda"):
        parser.error("this benchmark requires a CUDA device")
    tags = [spec.partition("=")[0] for spec in args.model]
    if len(tags) != len(set(tags)) or any(not spec.partition("=")[2] or not tag
                                         for spec, tag in zip(args.model, tags)):
        parser.error("models require unique nonempty TAG=MODEL labels")
    return args


def canonical_prompts(texts: list[str], count: int, chars: int) -> list[str]:
    if min(count, chars) < 1:
        raise ValueError("prompt count and character budget must be positive")
    prompts = []
    for text in texts:
        canonical = canonical_text(text, chars)
        if len(canonical) >= chars:
            prompts.append(canonical)
            if len(prompts) == count:
                return prompts
    raise ValueError(f"only {len(prompts)} eligible canonical prompts; need {count}")


def summarize_batches(records: list[dict]) -> dict:
    if not records:
        raise ValueError("no measured batches")
    fields = ["generation_seconds", "end_to_end_seconds", "prefill_seconds", "decode_seconds",
              "tokenization_seconds", "detokenization_seconds", "generated_tokens", "decode_tokens",
              "generated_chars", "input_tokens", "padded_input_tokens", "batch_size"]
    for record in records:
        if any(not math.isfinite(record[key]) or record[key] < 0 for key in fields):
            raise ValueError("measurements must be finite and nonnegative")
        if min(record["generation_seconds"], record["end_to_end_seconds"]) <= 0:
            raise ValueError("measured durations must be positive")
    totals = {key: sum(record[key] for record in records) for key in fields}
    for phase in ["generation", "end_to_end"]:
        for unit in ["tokens", "chars"]:
            totals[f"{phase}_{unit}_per_second"] = totals[f"generated_{unit}"] / totals[f"{phase}_seconds"]
    totals["decode_tokens_per_second"] = (totals["decode_tokens"] / totals["decode_seconds"]
                                          if totals["decode_seconds"] else None)
    for key in ["peak_allocated_bytes", "peak_reserved_bytes"]:
        totals[key] = max(record[key] for record in records)
    totals["n_prompts"] = totals.pop("batch_size")
    return {"n_batches": len(records), **totals}


@torch.inference_mode()
def run_batch(model, tokenizer, prompts: list[str], device: str, new_tokens: int) -> dict:
    if not prompts or new_tokens < 1:
        raise ValueError("nonempty prompts and positive generation length required")
    normalizer = tokenizer.backend_tokenizer.normalizer
    if normalizer and any(normalizer.normalize_str(text) != text for text in prompts):
        raise ValueError("tokenizer normalizer changes canonical prompts")
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    inputs = tokenizer(prompts, padding=True, truncation=False, add_special_tokens=False,
                       return_tensors="pt")
    tokenized = time.perf_counter()
    lengths = inputs.attention_mask.sum(-1).tolist()
    if min(lengths) < 1:
        raise ValueError("empty tokenized prompt")
    if inputs.input_ids.shape[1] + new_tokens > model.config.max_position_embeddings:
        raise ValueError("prompt plus requested generation exceeds model context")
    input_ids = inputs.input_ids.to(device)
    mask = inputs.attention_mask.to(device)
    positions = mask.cumsum(-1).sub(1).masked_fill(mask == 0, 0)
    torch.cuda.synchronize(device)
    generation_started = time.perf_counter()
    output = model(input_ids=input_ids, attention_mask=mask, position_ids=positions,
                   use_cache=True, logits_to_keep=1)
    next_ids = output.logits[:, -1].argmax(-1, keepdim=True)
    cache = output.past_key_values
    generated = [next_ids]
    del output
    torch.cuda.synchronize(device)
    prefilled = time.perf_counter()
    for _ in range(new_tokens - 1):
        mask = torch.cat([mask, torch.ones_like(next_ids)], dim=1)
        positions = mask.sum(-1, keepdim=True).sub(1)
        output = model(input_ids=next_ids, attention_mask=mask, position_ids=positions,
                       past_key_values=cache, use_cache=True, logits_to_keep=1)
        next_ids = output.logits[:, -1].argmax(-1, keepdim=True)
        cache = output.past_key_values
        generated.append(next_ids)
        del output
    torch.cuda.synchronize(device)
    generated_at = time.perf_counter()
    token_ids = torch.cat(generated, dim=1).cpu().tolist()
    texts = tokenizer.batch_decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    ended = time.perf_counter()
    return {"batch_size": len(prompts), "input_tokens": sum(lengths), "input_lengths": lengths,
            "padded_input_tokens": input_ids.numel(), "generated_tokens": sum(map(len, token_ids)),
            "decode_tokens": len(prompts) * (new_tokens - 1), "generated_chars": sum(map(len, texts)),
            "tokenization_seconds": tokenized - started, "detokenization_seconds": ended - generated_at,
            "prefill_seconds": prefilled - generation_started, "decode_seconds": generated_at - prefilled,
            "generation_seconds": generated_at - generation_started, "end_to_end_seconds": ended - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "generated_token_ids": token_ids, "generated_texts": texts}


def artifact_hashes(path: str, patterns: list[str] | None = None) -> dict:
    root = Path(path)
    if not root.is_dir():
        return {}
    files = set()
    for pattern in patterns or ["*.safetensors", "pytorch_model*.bin", "*.json", "*.model", "merges.txt", "vocab.txt"]:
        files.update(root.glob(pattern))
    hashes = {}
    for file in sorted(files):
        digest = hashlib.sha256()
        with file.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[file.name] = digest.hexdigest()
    return hashes


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)
    texts = load_texts(args.split, args.limit * 4, min_chars=args.prompt_chars)
    prompts = canonical_prompts(texts, args.limit, args.prompt_chars)
    properties = torch.cuda.get_device_properties(args.device)
    packages = {}
    for package in ["torch", "transformers", "tokenizers", "flash-attn", "accelerate"]:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(), "source": str(args.split.resolve()),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "protocol": {"normalizer": "paper_normalizer(preserve_alif=True), fixed point before character prefix", "padding_side": "left",
                     "do_sample": False, "stop_at_eos": False, "use_cache": True,
                     "dtype": "bfloat16", "add_special_tokens": False,
                     "prefill": "full prompt forward plus first-token argmax",
                     "decode": "remaining new_tokens-1 cached forwards and argmax",
                     "end_to_end": "CPU tokenization, H2D, generation, D2H and text decoding",
                     "memory": "CUDA peak allocator bytes including model, reset per batch after empty_cache",
                     "characters": "Unicode code points in decoded generated text, excluding special tokens"},
        "environment": {"hostname": socket.gethostname(), "platform": platform.platform(),
                        "python": platform.python_version(), "packages": packages,
                        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                        "gpu": properties.name, "gpu_total_bytes": properties.total_memory,
                        "gpu_capability": list(torch.cuda.get_device_capability(args.device)),
                        "nvidia_smi": subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout},
        "prompts": [{"index": i, "text": text, "chars": len(text),
                     "sha256": hashlib.sha256(text.encode()).hexdigest()} for i, text in enumerate(prompts)],
        "results": {},
    }
    for spec in args.model:
        tag, _, path = spec.partition("=")
        model_path, _, tokenizer_path = path.partition(":")
        tokenizer_path = tokenizer_path or model_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, padding_side="left")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None:
            raise ValueError("tokenizer requires a pad or EOS token")
        model = load_causal_lm(model_path, args.device).eval()
        result = {"model_path": model_path, "tokenizer_path": tokenizer_path,
                  "model_files_sha256": artifact_hashes(model_path),
                  "tokenizer_files_sha256": artifact_hashes(tokenizer_path),
                  "model_commit": getattr(model.config, "_commit_hash", None),
                  "attention_implementation": model.config._attn_implementation,
                  "model_config": model.config.to_dict(), "vocab_size": len(tokenizer), "batches": {}}
        for size in args.batch_sizes:
            for _ in range(args.warmup):
                run_batch(model, tokenizer, prompts[:size], args.device, args.new_tokens)
            records = []
            for index in range(0, len(prompts), size):
                record = run_batch(model, tokenizer, prompts[index:index + size], args.device, args.new_tokens)
                record["prompt_indices"] = list(range(index, min(index + size, len(prompts))))
                records.append(record)
            result["batches"][str(size)] = {"summary": summarize_batches(records), "measurements": records}
        report["results"][tag] = result
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        print(f"wrote {tag} to {args.out}", flush=True)
        del model
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
