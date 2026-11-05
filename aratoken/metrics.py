from __future__ import annotations

import hashlib
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


def per_doc_bpc(model, tok, texts, device, max_len: int = 1024,
                common_text_chars: int | None = None) -> tuple[np.ndarray, dict]:
    import torch

    documents = []
    if common_text_chars is not None:
        from aratoken.normalizer import canonical_text

        if common_text_chars < 1:
            raise ValueError("common_text_chars must be positive")
        normalizer = tok.backend_tokenizer.normalizer
        prefix_id = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
        if prefix_id is None:
            raise ValueError("Common-text scoring requires a BOS or EOS token")
        context_limit = min(max_len, model.config.max_position_embeddings)
    out, tot_nll, tot_chars, tot_tokens = [], 0.0, 0, 0
    with torch.no_grad():
        for index, text in enumerate(texts):
            if common_text_chars is None:
                ids = tok(text, return_tensors="pt", truncation=True,
                          max_length=max_len, add_special_tokens=False).input_ids.to(device)
                if ids.shape[1] < 2:
                    continue
                chars = len(tok.decode(ids[0], skip_special_tokens=True))
                tokens = ids.shape[1]
            else:
                canonical = canonical_text(text, common_text_chars)
                if not canonical:
                    raise ValueError(f"Document {index} has empty canonical text")
                if normalizer is not None and normalizer.normalize_str(canonical) != canonical:
                    raise ValueError(f"Tokenizer normalization changes canonical document {index}")
                target = tok(canonical, add_special_tokens=False, truncation=False).input_ids
                if not target:
                    raise ValueError(f"Document {index} has empty tokenization")
                if len(target) + 1 > context_limit:
                    raise ValueError(f"Document {index} exceeds context limit {context_limit}")
                ids = torch.tensor([[prefix_id, *target]], device=device)
                chars, tokens = len(canonical), len(target)
            nll = model(ids, labels=ids).loss.item() * (ids.shape[1] - 1)
            if chars == 0:
                continue
            out.append(nll / (chars * math.log(2)))
            tot_nll += nll
            tot_chars += chars
            tot_tokens += tokens
            if common_text_chars is not None:
                documents.append({"index": index, "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
                                  "text_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
                                  "chars": chars, "tokens": tokens, "nll": nll, "bpc": out[-1]})
    totals = {
        "corpus_bpc": tot_nll / (tot_chars * math.log(2)) if tot_chars else float("nan"),
        "total_chars": tot_chars,
        "total_tokens": tot_tokens,
        "chars_per_token": tot_chars / tot_tokens if tot_tokens else float("nan"),
    }
    if common_text_chars is not None:
        totals.update(documents=documents, protocol={"name": "common-text-bpc-v1",
                      "common_text_chars": common_text_chars, "character_unit": "Unicode code points",
                      "normalizer": "paper_normalizer(preserve_alif=True), fixed point before character prefix",
                      "truncation": "canonical character prefix; no token truncation",
                      "prefix_token_id": prefix_id, "prefix_token": tok.convert_ids_to_tokens(prefix_id),
                      "context_limit": context_limit, "scored_tokens": "all target tokens; prefix excluded"})
    return np.asarray(out), totals
