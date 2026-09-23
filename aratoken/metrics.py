from __future__ import annotations

import math

import numpy as np


def bootstrap_ci(values: np.ndarray, n_boot: int, seed: int, ci: float = 0.95) -> dict:
    rng = np.random.default_rng(seed)
    n = len(values)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = values[idx].mean(axis=1)
    a = (1 - ci) / 2
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "ci_lower": float(np.percentile(boot, a * 100)),
        "ci_upper": float(np.percentile(boot, (1 - a) * 100)),
        "ci_level": ci,
        "n_docs": int(n),
        "n_bootstrap": int(n_boot),
    }


def fertility_and_compression(encode, texts: list[str], n_boot: int, seed: int,
                              unk_id: int | None = None, oov_docs: int = 500) -> dict:
    fert, comp = [], []
    tot_tok = tot_word = tot_char = oov_words = oov_denom = 0
    for di, t in enumerate(texts):
        toks = t.split()
        words = len(toks)
        chars = len(t)
        if words == 0 or chars == 0:
            continue
        ntok = len(encode(t))
        if ntok == 0:
            continue
        if unk_id is not None and di < oov_docs:
            oov_words += sum(1 for w in toks if unk_id in encode(w))
            oov_denom += words
        fert.append(ntok / words)
        comp.append(chars / ntok)
        tot_tok += ntok
        tot_word += words
        tot_char += chars
    fert = np.asarray(fert)
    comp = np.asarray(comp)
    return {
        "macro": {"fertility": bootstrap_ci(fert, n_boot, seed),
                  "compression": bootstrap_ci(comp, n_boot, seed)},
        "micro": {"fertility": tot_tok / tot_word,
                  "compression": tot_char / tot_tok},
        "oov_rate_percent": (100.0 * oov_words / oov_denom) if
                            (unk_id is not None and oov_denom) else 0.0,
        "oov_words_scored": oov_denom,
        "totals": {"tokens": tot_tok, "words": tot_word, "chars": tot_char,
                   "docs": int(len(fert))},
    }


def per_doc_bpc(model, tok, texts, device, max_len: int = 1024) -> tuple[np.ndarray, dict]:
    import torch

    out, tot_nll, tot_chars, tot_tokens = [], 0.0, 0, 0
    with torch.no_grad():
        for text in texts:
            ids = tok(text, return_tensors="pt", truncation=True,
                      max_length=max_len, add_special_tokens=False).input_ids.to(device)
            if ids.shape[1] < 2:
                continue
            nll = model(ids, labels=ids).loss.item() * (ids.shape[1] - 1)
            chars = len(tok.decode(ids[0], skip_special_tokens=True))
            if chars == 0:
                continue
            out.append(nll / (chars * math.log(2)))
            tot_nll += nll
            tot_chars += chars
            tot_tokens += ids.shape[1]
    return np.asarray(out), {
        "corpus_bpc": tot_nll / (tot_chars * math.log(2)) if tot_chars else float("nan"),
        "total_chars": tot_chars,
        "total_tokens": tot_tokens,
        "chars_per_token": tot_chars / tot_tokens if tot_tokens else float("nan"),
    }
