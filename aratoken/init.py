from __future__ import annotations

from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

METHODS = ("random", "mean", "calib", "wechsel", "focus", "hybrid")


def char_ngrams(s: str, n_min: int = 2, n_max: int = 4) -> Counter:
    s = s.strip()
    c: Counter = Counter()
    for n in range(n_min, n_max + 1):
        for i in range(len(s) - n + 1):
            c[s[i:i + n]] += 1
    if not c:
        c[s] = 1
    return c


def compositional(emb: torch.Tensor, base_tok, surface: str) -> torch.Tensor | None:
    ids = base_tok(surface, add_special_tokens=False)["input_ids"]
    return emb[torch.tensor(ids)].mean(0) if ids else None


def init_random(emb, added, sigma):
    for i in added:
        emb[i].normal_(0.0, sigma)


def init_mean(emb, added, base_tok, sigma, old_n, seed, moment_matched: bool):
    mu = emb[:old_n].mean(0)
    radii = (emb[:old_n] - mu).norm(dim=1)
    g = torch.Generator().manual_seed(seed)
    for i, surf in added.items():
        ids = base_tok(surf, add_special_tokens=False)["input_ids"]
        if not ids and not moment_matched:
            continue
        if not ids:
            emb[i] = torch.empty_like(emb[i]).normal_(0.0, sigma)
            continue
        rows = emb[torch.tensor(ids)]
        if not moment_matched:
            emb[i] = rows.mean(0)
            continue
        d = (rows - mu).mean(0)
        n = d.norm()
        if n < 1e-6:
            noise = torch.empty_like(d).normal_(0.0, 1.0, generator=g)
            emb[i] = mu + noise.mul_(radii[torch.randint(len(radii), (1,), generator=g)] / (d.numel() ** 0.5))
            continue
        r = radii[torch.randint(len(radii), (1,), generator=g)]
        emb[i] = mu + d * (r / n)


def init_analogical(emb, added, base_tok, new_tok, sigma, method, topk, alpha):
    base_vocab = base_tok.get_vocab()
    src_tokens = list(base_vocab)
    if method == "focus":
        new_vocab = new_tok.get_vocab()
        src_tokens = [t for t in base_vocab if t in new_vocab] or src_tokens
    src_ids = torch.tensor([base_vocab[t] for t in src_tokens])
    gram_index: dict[str, list[int]] = {}
    for j, t in enumerate(src_tokens):
        for k in char_ngrams(base_tok.convert_tokens_to_string([t])):
            gram_index.setdefault(k, []).append(j)

    src_emb = emb[src_ids]
    for i, surf in added.items():
        cand: Counter = Counter()
        for k, v in char_ngrams(surf).items():
            for j in gram_index.get(k, ())[:400]:
                cand[j] += v
        comp = compositional(emb, base_tok, surf)
        if not cand:
            emb[i] = comp if comp is not None else torch.empty_like(emb[i]).normal_(0.0, sigma)
            continue
        best = cand.most_common(topk)
        w = torch.tensor([b[1] for b in best], dtype=torch.float32)
        analog = (src_emb[torch.tensor([b[0] for b in best])] * (w / w.sum()).unsqueeze(1)).sum(0)
        if method == "hybrid":
            emb[i] = alpha * (comp if comp is not None else analog) + (1.0 - alpha) * analog
        else:
            emb[i] = analog


@torch.no_grad()
def build(method: str, tok_path: str, out_path: Path, base: str,
          topk: int = 10, seed: int = 42, alpha: float = 0.5) -> None:
    torch.manual_seed(seed)
    base_tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    new_tok = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    dtype = torch.bfloat16 if method == "mean" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype, trust_remote_code=True)

    old_n = model.get_input_embeddings().weight.shape[0]
    start = len(base_tok) if method == "mean" else old_n
    model.resize_token_embeddings(len(new_tok))
    emb = model.get_input_embeddings().weight
    sigma = emb[:old_n].float().std().item()
    added = {i: t for t, i in new_tok.get_vocab().items() if i >= start}
    print(f"{method}: {old_n} -> {len(new_tok)} rows, {len(added)} initialized, sigma={sigma:.4f}")

    if method == "random":
        init_random(emb, added, sigma)
    elif method in ("mean", "calib"):
        init_mean(emb, added, base_tok, sigma, old_n, seed, moment_matched=method == "calib")
    elif method in ("wechsel", "focus", "hybrid"):
        init_analogical(emb, added, base_tok, new_tok, sigma, method, topk, alpha)
    else:
        raise SystemExit(f"unknown method {method}")

    out_emb = model.get_output_embeddings()
    if out_emb is not None and out_emb.weight.data_ptr() != emb.data_ptr():
        out_emb.weight[start:] = emb[start:]

    model.config.lep_added_tokens_start = start
    model.config.lep_init_method = method
    out_path.mkdir(parents=True, exist_ok=True)
    model.to(torch.bfloat16).save_pretrained(str(out_path))
    print(f"wrote {out_path}")
