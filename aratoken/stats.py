from __future__ import annotations

import numpy as np


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int, seed: int) -> dict:
    assert len(a) == len(b), (len(a), len(b))
    rng = np.random.default_rng(seed)
    d = a - b
    n = len(d)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = d[idx].mean(axis=1)
    obs = float(d.mean())
    centred = boot - boot.mean()
    return {
        "mean_diff": obs,
        "ci_lower": float(np.percentile(boot, 2.5)),
        "ci_upper": float(np.percentile(boot, 97.5)),
        "p_value": float((np.abs(centred) >= abs(obs)).mean()),
        "n_items": int(n),
        "n_bootstrap": int(n_boot),
    }


def paired_bootstrap_metric(y_true, a_pred, b_pred, metric, n_boot: int, seed: int,
                            name: str = "delta_macro_f1") -> dict:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    obs = metric(y_true, a_pred) - metric(y_true, b_pred)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = metric(y_true[idx], a_pred[idx]) - metric(y_true[idx], b_pred[idx])
    centred = diffs - diffs.mean()
    return {name: float(obs),
            "ci_lower": float(np.percentile(diffs, 2.5)),
            "ci_upper": float(np.percentile(diffs, 97.5)),
            "p_value": float((np.abs(centred) >= abs(obs)).mean()),
            "n_items": int(n), "n_bootstrap": int(n_boot)}


def accuracy(t, p) -> float:
    return float(np.mean(np.asarray(t) == np.asarray(p)))


def holm(pvals: dict[str, float], alpha: float = 0.05) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(running, (m - i) * p))
        running = adj
        out[k] = {"p_raw": p, "p_holm": adj, "reject_at_0.05": adj < alpha}
    return out
