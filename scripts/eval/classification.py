import argparse
import json
import random
from pathlib import Path

import torch
from datasets import load_dataset
from sklearn.metrics import f1_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

LETTERS = ["أ", "ب", "ج", "د", "هـ"]


def sentiment_header(labels):
    return f"صنف مشاعر النص العربي التالي كواحدة مما يلي: {', '.join(labels)}.\n"


def sarcasm_header(labels):
    return '"أُعطيت لك تغريدة باللغة العربية. حدد ما إذا كانت التغريدة ساخرة أم لا. أجب بـ "ساخرة" أو "غير ساخرة"\n'


def news_header(labels):
    return (f"صنف موضوع المقال الإخباري العربي التالي إلى أحد المواضيع التالية: {', '.join(labels)}. "
            "أجب باسم فئة واحدة فقط من القائمة أعلاه.")


def xnli_header(labels):
    return ("حدِّد العلاقة بين المقدّمة والفرضية المكتوبتين بالعربية.\n"
            f"اختر أحد الخيارات التالية: {', '.join(labels)}. أجب بكلمة واحدة فقط.\n")


DATASETS = {
    "ASTD": dict(path="arbml/ASTD", split="train", test_fraction=0.20, text_col="tweet", label_col="label",
                 labels={0: "حيادي", 1: "موضوعي", 2: "إيجابي", 3: "سلبي"}, header=sentiment_header,
                 text_name="النص", label_name="التسمية"),
    "ArSarcasm": dict(path="iabufarha/ar_sarcasm", split="test", text_col="tweet", label_col="sentiment",
                      labels={0: "سلبي", 1: "حيادي", 2: "إيجابي"}, header=sentiment_header,
                      text_name="النص", label_name="التقييم"),
    "ArSarcasm-Sarcasm": dict(path="iabufarha/ar_sarcasm", split="test", text_col="tweet", label_col="sarcasm",
                              labels={0: "غير ساخرة", 1: "ساخرة"}, header=sarcasm_header,
                              text_name="تغريدة", label_name="التقييم"),
    "sanad": dict(path="arbml/SANAD", split="train", test_fraction=0.05, text_col="Article", label_col="label",
                  labels={0: "التكنولوجيا", 1: "المالية", 2: "السياسة", 3: "الدين", 4: "الطب", 5: "الثقافة",
                          6: "الرياضة"}, header=news_header, text_name="مقالة", label_name="التقييم"),
    "xnli": dict(path="facebook/xnli", config="ar", split="validation", premise_col="premise",
                 hypothesis_col="hypothesis", label_col="label", labels={0: "استلزام", 1: "محايد", 2: "تناقض"},
                 header=xnli_header),
}


def eval_split(cfg: dict, seed: int):
    if "config" in cfg:
        return load_dataset(cfg["path"], cfg["config"], split=cfg["split"])
    ds = load_dataset(cfg["path"], split=cfg["split"])
    if "test_fraction" not in cfg:
        return ds
    return ds.train_test_split(test_size=cfg["test_fraction"], seed=seed)["test"]


def item(cfg: dict, rec: dict):
    if "premise_col" in cfg:
        return rec[cfg["premise_col"]], rec[cfg["hypothesis_col"]]
    return rec[cfg["text_col"]]


def pick_demos(ds, cfg: dict, k: int):
    idxs = list(range(len(ds)))
    random.shuffle(idxs)
    first_of_label = {}
    for i in idxs:
        first_of_label.setdefault(cfg["labels"][ds[i][cfg["label_col"]]], i)
        if len(first_of_label) == k:
            break
    chosen = list(first_of_label.values())
    chosen += [i for i in idxs if i not in first_of_label.values()][:k - len(chosen)]
    return [(item(cfg, ds[i]), cfg["labels"][ds[i][cfg["label_col"]]]) for i in chosen], set(chosen)


def prompt_body(cfg: dict, text, labels, demos) -> str:
    lines = [cfg["header"](labels)]
    if "premise_col" in cfg:
        for (prem, hyp), lbl in demos:
            lines.append(f'المقدّمة: "{prem}"\nالفرضية: "{hyp}"\nالتقييم: {lbl}\n')
        lines.append(f'المقدّمة: "{text[0]}"\nالفرضية: "{text[1]}"\nالتقييم:')
    else:
        for t, lbl in demos:
            lines.append(f'{cfg["text_name"]}: "{t}"\n{cfg["label_name"]}: {lbl}\n')
        lines.append(f'{cfg["text_name"]}: "{text}"\n{cfg["label_name"]}:')
    return "\n".join(lines)


def mcq_prompt(cfg: dict, text, demos, rotate: int) -> tuple[str, list[str], list[str]]:
    labels = list(cfg["labels"].values())
    r = rotate % len(labels)
    order = labels[r:] + labels[:r]
    letters = LETTERS[:len(labels)]
    by_letter = dict(zip(order, LETTERS))
    head, *rest = prompt_body(cfg, text, labels, [(t, by_letter.get(lbl, lbl)) for t, lbl in demos]).split("\n")
    options = "\n".join(f"{c}. {lbl}" for c, lbl in zip(letters, order))
    return "\n".join([head, options, *rest]), letters, order


@torch.no_grad()
def predict(model, tok, cfg, text, demos, rotate: int, device) -> str:
    prompt, letters, order = mcq_prompt(cfg, text, demos, rotate)
    prompt_ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    scores = []
    for letter in letters:
        lab_ids = tok(" " + letter, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        ids = torch.cat([prompt_ids, lab_ids], dim=1)
        logprobs = torch.log_softmax(model(ids).logits[:, :-1].float(), dim=-1)
        tok_lp = logprobs.gather(2, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        scores.append(tok_lp[0, -lab_ids.shape[1]:].sum().item() / len((" " + letter).encode("utf-8")))
    return order[max(range(len(letters)), key=scores.__getitem__)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--datasets", nargs="+", default=list(DATASETS))
    p.add_argument("--shots", type=int, default=0, choices=[0, 3])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, device_map=a.device)
    a.out.mkdir(parents=True, exist_ok=True)
    for name in a.datasets:
        random.seed(a.seed)
        torch.manual_seed(a.seed)
        cfg = DATASETS[name]
        ds = eval_split(cfg, a.seed)
        demos, skip = pick_demos(ds, cfg, a.shots) if a.shots else ([], set())
        y_true, y_pred = [], []
        for i, rec in enumerate(tqdm(ds, desc=f"{name} {a.shots}-shot")):
            if i in skip:
                continue
            y_true.append(cfg["labels"][rec[cfg["label_col"]]])
            y_pred.append(predict(model, tok, cfg, item(cfg, rec), demos, len(y_pred), a.device))
        labels = list(cfg["labels"].values())
        print(f"{name} {a.shots}-shot macro F1 {f1_score(y_true, y_pred, labels=labels, average='macro'):.3f}")
        (a.out / f"{name}_{a.shots}shot_s{a.seed}.json").write_text(json.dumps(
            {"dataset": name, "shots": a.shots, "seed": a.seed, "labels": labels,
             "y_true": y_true, "y_pred": y_pred}, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
