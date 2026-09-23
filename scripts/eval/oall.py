import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = "السؤال: {q}\nالإجابة:"
CONTEXT = "{c}\nالسؤال: {q}\nالإجابة:"
LETTERS = ["أ", "ب", "ج", "د", "هـ"]
EN_LETTERS = ["A", "B", "C", "D", "E"]
EN_PROMPT = "Question: {q}\nAnswer:"
MCQ = "السؤال: {q}\n{opts}\nالإجابة:"


def _mbzuai(rec: dict) -> tuple[str, list[str], int] | None:
    opts = [rec[f"Option {i}"] for i in range(1, 6)]
    opts = [o for o in opts if o and o != "None"]
    gold = "ABCDE".find(rec["Answer Key"])
    if gold < 0 or gold >= len(opts):
        return None
    ctx = rec.get("Context")
    q = (CONTEXT.format(c=ctx, q=rec["Question"]) if ctx and ctx != "None"
         else PROMPT.format(q=rec["Question"]))
    return q, opts, gold


def _letters(rec: dict, keys: str, answer_col: str) -> tuple[str, list[str], int] | None:
    opts = [rec[k] for k in keys if rec.get(k)]
    gold = keys.find(str(rec[answer_col]).strip()[:1])
    if gold < 0 or gold >= len(opts):
        return None
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


def as_mcq(prompt: str, choices: list[str], letters: list[str],
           rotate: int = 0) -> tuple[str, list[str], list[int]]:
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
    tok = AutoTokenizer.from_pretrained(a.tokenizer or a.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, trust_remote_code=True).to(a.device).eval()

    a.out = a.out / a.protocol / a.lang if a.lang == "en" else a.out / a.protocol
    a.out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, float] = {}
    letters = EN_LETTERS if a.lang == "en" else LETTERS
    for task, repo, configs, split, adapt in (EN_TASKS if a.lang == "en" else TASKS):
        done = a.out / f"{a.tag}_{task}.json"
        if done.exists():
            summary[task] = json.loads(done.read_text())["accuracy"]
            print(f"  {task:14} acc {summary[task]:.4f}  (cached)", flush=True)
            continue
        y_true, y_pred = [], []
        for cfg in configs:
            ds = load_dataset(repo, cfg, split=split)
            if a.limit:
                ds = ds.select(range(min(a.limit, len(ds))))
            for rec in tqdm(ds, desc=f"{a.tag} {task}/{cfg}"):
                if repo.endswith("AraTrust-categorized"):
                    rec = dict(rec)
                    rec["Answer"] = ARABIC_LETTERS.get(
                        str(rec["Answer"]).strip()[:1], rec["Answer"])
                item = adapt(rec)
                if item is None:
                    continue
                prompt, choices, gold = item
                order = list(range(len(choices)))
                if a.protocol == "mcq":
                    prompt, choices, order = as_mcq(prompt, choices, letters,
                                                    rotate=len(y_true))
                y_true.append(gold)
                pick = best_choice(prompt, choices, tok, model, a.device)
                y_pred.append(order[pick])
        acc = sum(t == p for t, p in zip(y_true, y_pred)) / max(len(y_true), 1)
        summary[task] = acc
        (a.out / f"{a.tag}_{task}.json").write_text(
            json.dumps({"task": task, "model": a.model, "accuracy": acc,
                        "n": len(y_true), "y_true": y_true, "y_pred": y_pred}),
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
