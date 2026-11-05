import argparse
import hashlib
import json
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.eval.classification import LETTERS
from scripts.eval.inference import artifact_hashes

PROMPT = "السؤال: {q}\nالإجابة:"
CONTEXT = "{c}\nالسؤال: {q}\nالإجابة:"
EN_LETTERS = ["A", "B", "C", "D", "E"]
EN_PROMPT = "Question: {q}\nAnswer:"
MCQ = "السؤال: {q}\n{opts}\nالإجابة:"


def _mbzuai(rec: dict) -> tuple[str, list[str], int] | None:
    options = {key: rec[f"Option {i}"] for i, key in enumerate("ABCDE", 1)}
    keys = [key for key, value in options.items() if value and value != "None"]
    answer = str(rec["Answer Key"]).strip()
    if answer not in keys:
        return None
    opts = [options[key] for key in keys]
    gold = keys.index(answer)
    ctx = rec.get("Context")
    q = (CONTEXT.format(c=ctx, q=rec["Question"]) if ctx and ctx != "None"
         else PROMPT.format(q=rec["Question"]))
    return q, opts, gold


def _letters(rec: dict, keys: str, answer_col: str) -> tuple[str, list[str], int] | None:
    valid_keys = [key for key in keys if rec.get(key)]
    answer = str(rec[answer_col]).strip()[:1]
    if answer not in valid_keys:
        return None
    opts = [rec[key] for key in valid_keys]
    gold = valid_keys.index(answer)
    return PROMPT.format(q=rec["Question" if "Question" in rec else "question"]), opts, gold


def _mmlu(rec: dict) -> tuple[str, list[str], int] | None:
    ch = list(rec["choices"])
    g = int(rec["answer"])
    if g >= len(ch):
        return None
    return EN_PROMPT.format(q=rec["question"]), ch, g


def _arc(rec: dict) -> tuple[str, list[str], int] | None:
    ch = list(rec["choices"]["text"])
    labels = list(rec["choices"]["label"])
    if rec["answerKey"] not in labels:
        return None
    return EN_PROMPT.format(q=rec["question"]), ch, labels.index(rec["answerKey"])


def _alghafa(rec: dict) -> tuple[str, list[str], int] | None:
    opts = [rec[f"sol{i}"] for i in range(1, 5) if rec.get(f"sol{i}")]
    gold = int(rec["label"])
    if gold >= len(opts):
        return None
    return PROMPT.format(q=rec["query"]), opts, gold


TASKS = [
    ("ArabicMMLU", "MBZUAI/ArabicMMLU", ["All"], "test", _mbzuai),
    ("MadinahQA", "MBZUAI/MadinahQA",
     ["Arabic Language (General)", "Arabic Language (Grammar)"], "test", _mbzuai),
    ("ArabicEXAMS", "OALL/Arabic_EXAMS", ["default"], "test",
     lambda r: _letters(r, "ABCD", "answer")),
    ("AraTrust", "asas-ai/AraTrust-categorized",
     ["Ethics", "Illegal", "MentalHealth", "Offensive", "PhysicalHealth", "Privacy"],
     "train", lambda r: _letters(r, "ABC", "Answer")),
    ("AlGhafa", "OALL/AlGhafa-Arabic-LLM-Benchmark-Native",
     ["mcq_exams_test_ar", "meta_ar_msa", "multiple_choice_facts_truefalse_balanced_task",
      "multiple_choice_grounded_statement_soqal_task",
      "multiple_choice_grounded_statement_xglue_mlqa_task"], "test", _alghafa),
]

EN_TASKS = [
    ("MMLU", "cais/mmlu", ["all"], "test", _mmlu),
    ("ARC-Easy", "allenai/ai2_arc", ["ARC-Easy"], "test", _arc),
]

ARABIC_LETTERS = {"أ": "A", "ب": "B", "ج": "C"}


def json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def resolve_artifact(path: str) -> Path:
    return Path(path if Path(path).exists() else snapshot_download(path, local_files_only=True)).resolve()


def artifact_hash(path: Path, kind: str) -> str:
    patterns = (["config.json", "*.safetensors", "pytorch_model*.bin", "*.index.json"]
                if kind == "model" else ["tokenizer*", "special_tokens_map.json", "added_tokens.json",
                                         "vocab.*", "merges.txt", "spiece.model"])
    hashes = artifact_hashes(str(path), patterns)
    if not hashes:
        raise ValueError(f"No {kind} files found in {path}")
    return json_hash(hashes)


def task_items(repo, configs, split, adapt, limit):
    items, excluded, sources = [], [], []
    for cfg in configs:
        ds = load_dataset(repo, cfg, split=split)
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        sources.append({"repo": repo, "config": cfg, "split": split,
                        "fingerprint": ds._fingerprint, "selected_rows": len(ds),
                        "cache_files": ds.cache_files})
        for index, rec in enumerate(ds):
            identity = {"repo": repo, "config": cfg, "split": split,
                        "row_index": index, "source_sha256": json_hash(rec)}
            if repo.endswith("AraTrust-categorized"):
                rec = dict(rec)
                rec["Answer"] = ARABIC_LETTERS.get(str(rec["Answer"]).strip()[:1], rec["Answer"])
            item = adapt(rec)
            if item is None:
                excluded.append({**identity, "reason": "adapter rejected invalid or missing gold"})
                continue
            prompt, choices, gold = item
            accepted = {**identity, "prompt": prompt, "choices": choices, "gold": gold}
            items.append({**accepted, "item_sha256": json_hash(accepted)})
    return items, excluded, sources


def evaluation_protocol(lang: str, limit: int, mode: str) -> dict:
    paths = [Path(__file__).with_name(name) for name in ("oall.py", "classification.py", "inference.py")]
    return {"name": "continuation-v2", "lang": lang, "limit": limit,
            "dtype": "bfloat16", "mode": mode,
            "rotation": "accepted item index modulo options" if mode == "mcq" else "none",
            "score": "sum log probability / raw UTF-8 bytes including leading space",
            "source_sha256": json_hash({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})}


def as_mcq(prompt: str, choices: list[str], letters: list[str],
           rotate: int = 0) -> tuple[str, list[str], list[int]]:
    if not 1 <= len(choices) <= len(letters):
        raise ValueError(f"unsupported choice count: {len(choices)}")
    body = prompt.split("\n")
    stem = "\n".join(body[:-1]) if len(body) > 1 else body[0]
    tail = "الإجابة:" if letters is LETTERS else "Answer:"
    r = rotate % len(choices)
    order = list(range(r, len(choices))) + list(range(r))
    opts = "\n".join(f"{letters[i]}. {choices[j].strip()}"
                     for i, j in enumerate(order))
    return f"{stem}\n{opts}\n{tail}", letters[:len(choices)], order


@torch.no_grad()
def best_choice(prompt: str, choices: list[str], tok, model, device) -> int:
    p_ids = tok(prompt, add_special_tokens=False).input_ids
    scores = []
    for c in choices:
        c_ids = tok(" " + c.strip(), add_special_tokens=False).input_ids
        if not c_ids:
            scores.append(-1e9)
            continue
        ids = torch.tensor([p_ids + c_ids], device=device)
        logits = model(ids).logits[0, :-1].float().log_softmax(-1)
        tgt = ids[0, 1:]
        lp = logits[torch.arange(len(tgt), device=device), tgt][len(p_ids) - 1:]
        cont = " " + c.strip()
        scores.append(lp.sum().item() / max(len(cont.encode("utf-8")), 1))
    return int(max(range(len(scores)), key=scores.__getitem__))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--tag", required=True)
    p.add_argument("--out", type=Path, default=Path("results/oall"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--lang", choices=["ar", "en"], default="ar",
                   help="ar runs the OALL v2 suite; en runs the English "
                        "retention probe under the same protocol.")
    p.add_argument("--protocol", choices=["cloze", "mcq"], default="mcq",
                   help="cloze scores the answer strings as continuations; mcq "
                        "lists them in the prompt and scores the ordinal letter.")
    p.add_argument("--limit", type=int, default=1000,
                   help="Cap per configuration, so ArabicMMLU does not "
                        "dominate the suite's wall clock.")
    return p.parse_args()


def main() -> int:
    a = parse_args()
    model_path, tokenizer_path = resolve_artifact(a.model), resolve_artifact(a.tokenizer or a.model)
    model_digest, tokenizer_digest = artifact_hash(model_path, "model"), artifact_hash(tokenizer_path, "tokenizer")
    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).to(a.device).eval()

    a.out = a.out / a.protocol / a.lang if a.lang == "en" else a.out / a.protocol
    a.out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, float] = {}
    letters = EN_LETTERS if a.lang == "en" else LETTERS
    protocol = evaluation_protocol(a.lang, a.limit, a.protocol)
    for task, repo, configs, split, adapt in (EN_TASKS if a.lang == "en" else TASKS):
        items, excluded, sources = task_items(repo, configs, split, adapt, a.limit)
        if not items:
            raise ValueError(f"No valid items for {task}")
        hashes = [item["item_sha256"] for item in items]
        golds = [item["gold"] for item in items]
        receipt = {"task": task, "model": str(model_path), "tokenizer": str(tokenizer_path),
                   "model_sha256": model_digest, "tokenizer_sha256": tokenizer_digest,
                   "n": len(items), "item_sha256": hashes, "cohort_sha256": json_hash(hashes),
                   "protocol": protocol}
        done = a.out / f"{a.tag}_{task}.json"
        if done.exists():
            cached = json.loads(done.read_text())
            predictions = cached.get("y_pred", [])
            if (any(cached.get(k) != v for k, v in receipt.items()) or cached.get("y_true") != golds
                    or len(predictions) != len(items)
                    or any(type(p) is not int or not 0 <= p < len(item["choices"])
                           for p, item in zip(predictions, items))
                    or cached.get("accuracy") != sum(t == p for t, p in zip(golds, predictions)) / len(items)):
                raise ValueError(f"Invalid or stale evaluation cache: {done}")
            summary[task] = cached["accuracy"]
            print(f"  {task:14} acc {summary[task]:.4f}  (cached)", flush=True)
            continue
        y_true, y_pred = [], []
        for item in tqdm(items, desc=f"{a.tag} {task}"):
            prompt, choices, gold = item["prompt"], item["choices"], item["gold"]
            order = list(range(len(choices)))
            if a.protocol == "mcq":
                prompt, choices, order = as_mcq(prompt, choices, letters, rotate=len(y_true))
            y_true.append(gold)
            pick = best_choice(prompt, choices, tok, model, a.device)
            y_pred.append(order[pick])
        acc = sum(t == p for t, p in zip(y_true, y_pred)) / max(len(y_true), 1)
        summary[task] = acc
        (a.out / f"{a.tag}_{task}.json").write_text(
            json.dumps({**receipt, "accuracy": acc, "y_true": y_true, "y_pred": y_pred,
                        "excluded": excluded, "data_sources": sources}),
            encoding="utf-8")
        print(f"  {task:14} acc {acc:.4f}  (n={len(y_true)})", flush=True)

    summary["macro"] = sum(summary.values()) / len(summary)
    (a.out / f"{a.tag}_summary.json").write_text(
        json.dumps({"model": a.model, "tag": a.tag, "accuracy": summary}, indent=2),
        encoding="utf-8")
    print(f"\n{a.tag}: macro {summary['macro']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
