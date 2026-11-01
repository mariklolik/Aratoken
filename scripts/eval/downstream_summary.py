import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from aratoken.stats import holm
from scripts.eval.rebuttal_summary import crossed_bootstrap, read_evaluation, seed_summary

TASKS = {'ar': ('ArabicMMLU', 'MadinahQA', 'ArabicEXAMS', 'AraTrust', 'AlGhafa'),
         'en': ('MMLU', 'ARC-Easy')}
MODELS = {'b06': None, **{f'{tag}500t{s}': f'budget500m_{arm}_s{s}'
                          for tag, arm in (('l', 'hybrid'), ('c', 'cpt')) for s in (1, 2, 3)}}
PAIRS = {'lep-cpt': ('l', 'c'), 'lep-base': ('l', 'b'), 'cpt-base': ('c', 'b')}


def task_bootstrap(differences: list[np.ndarray], n_bootstrap: int = 10000, seed: int = 42) -> dict:
    arrays = [np.asarray(d, dtype=float) for d in differences]
    if not arrays or n_bootstrap < 1 or any(d.ndim != 2 or d.shape[0] != 3 or not d.shape[1]
                                           or not np.isfinite(d).all() for d in arrays):
        raise ValueError('Three paired seeds and nonempty finite task arrays required')
    rng = np.random.default_rng(seed)
    boot = np.empty(n_bootstrap)
    for start in range(0, n_bootstrap, 128):
        count = min(128, n_bootstrap - start)
        seeds = rng.integers(0, 3, size=(count, 3))
        batch = np.zeros(count)
        for d in arrays:
            items = rng.integers(0, d.shape[1], size=(count, d.shape[1]))
            batch += d[seeds[:, :, None], items[:, None, :]].mean(axis=(1, 2)) / len(arrays)
        boot[start:start + count] = batch
    observed = float(np.mean([d.mean() for d in arrays]))
    exceedances = np.count_nonzero(np.abs(boot - boot.mean()) >= abs(observed))
    return {'mean_diff': observed, 'ci_lower': float(np.percentile(boot, 2.5)),
            'ci_upper': float(np.percentile(boot, 97.5)),
            'p_value': float((exceedances + 1) / (n_bootstrap + 1)),
            'n_seeds': 3, 'n_items_by_task': [d.shape[1] for d in arrays],
            'n_bootstrap': n_bootstrap, 'rng_seed': seed,
            'resampling': 'shared paired training seeds; paired items within each fixed task'}


def digest_valid(value: str) -> bool:
    return isinstance(value, str) and len(value) == 64 and not set(value) - set('0123456789abcdef')


def cohort_digest(hashes: list[str]) -> str:
    return hashlib.sha256(json.dumps(hashes, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_task(path: Path, task: str, frozen: dict, protocol: dict, model: Path,
              tokenizer_sha256: str, model_sha256: str | None) -> tuple[dict, dict]:
    data = json.loads(path.read_text())
    count, hashes = frozen['n'], data['item_sha256']
    if (data['task'] != task or any(data[k] != frozen[k] for k in ('n', 'item_sha256', 'cohort_sha256', 'y_true'))
            or len(data['y_pred']) != count or data['protocol'] != protocol):
        raise ValueError('Task count, ordered hashes, cohort, gold or protocol differs from frozen preflight')
    if (Path(data['model']).resolve() != model.resolve() or not digest_valid(data['model_sha256'])
            or (model_sha256 is not None and data['model_sha256'] != model_sha256)
            or data['tokenizer_sha256'] != tokenizer_sha256):
        raise ValueError('Model path/digest or tokenizer digest differs from registered identity')
    gold, predictions = np.asarray(data['y_true']), np.asarray(data['y_pred'])
    if any(v.ndim != 1 or not np.issubdtype(v.dtype, np.integer) or np.any(v < 0)
           for v in (gold, predictions)):
        raise ValueError('Task labels must be nonnegative integer vectors')
    values = (gold == predictions).astype(float) * 100
    signature = (hashes, data['cohort_sha256'], data['protocol'], data['y_true'])
    return {'status': 'complete', 'accuracy': float(values.mean()), 'n': count, 'model': data['model'],
            'model_sha256': data['model_sha256'], 'tokenizer_sha256': data['tokenizer_sha256'],
            'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'cohort_sha256': data['cohort_sha256'], 'protocol': data['protocol']}, {
                'values': values, 'signature': signature}


def load_models(root: Path, cohort: dict, expected_bpc_docs: int) -> tuple[dict, dict]:
    reports, cached = {}, {}
    for tag, run in MODELS.items():
        model = root / 'runs' / run / 'model/final_model' if run else Path(cohort['base_model']['path'])
        model_digest = None if run else cohort['base_model']['sha256']
        tokenizer_digest = cohort['tokenizer_sha256']['extended' if tag.startswith('l') else 'base']
        report = {'run': run, 'training_seed': int(tag[-1]) if run else None,
                  'tasks': {}, 'macro_accuracy': {}, 'pooled_accuracy': {}}
        cache = {}
        for language, tasks in TASKS.items():
            report['tasks'][language], cache[language] = {}, {}
            directory = root / 'downstream-v1/mcq' / ('en' if language == 'en' else '')
            for task in tasks:
                path = directory / f'{tag}_{task}.json'
                try:
                    entry, data = read_task(path, task, cohort['tasks'][language][task],
                                            cohort['protocols'][language], model, tokenizer_digest, model_digest)
                    model_digest = entry['model_sha256']
                    cache[language][task] = data
                except FileNotFoundError:
                    entry = {'status': 'pending', 'path': str(path)}
                except (ValueError, KeyError, TypeError) as error:
                    entry = {'status': 'invalid', 'path': str(path), 'reason': str(error)}
                report['tasks'][language][task] = entry
            if len(cache[language]) == len(tasks):
                report['macro_accuracy'][language] = float(np.mean([v['values'].mean() for v in cache[language].values()]))
                report['pooled_accuracy'][language] = float(np.concatenate([v['values'] for v in cache[language].values()]).mean())
        path = root / 'runs' / run / 'bpc_en.json' if run else root / 'base06_common_en.json'
        try:
            summary, data = read_evaluation(path, run or 'base', expected_bpc_docs)
            cache['en_bpc'] = {'values': data['values'], 'signature': (
                [(d['source_sha256'], d['text_sha256'], d['chars']) for d in data['documents']], data['protocol'])}
            report['en_bpc'] = {'status': 'complete', **summary}
        except FileNotFoundError:
            report['en_bpc'] = {'status': 'pending', 'path': str(path)}
        except (ValueError, KeyError, TypeError) as error:
            report['en_bpc'] = {'status': 'invalid', 'path': str(path), 'reason': str(error)}
        reports[tag], cached[tag] = report, cache
    return reports, cached


def paired_tasks(endpoint: str, left: str, right: str, reports: dict, cached: dict) -> tuple[dict, list]:
    language = endpoint.split('_')[0]
    tasks = TASKS[language] if endpoint.endswith('accuracy') else ('en_bpc',)
    differences, problems = [], []
    for task in tasks:
        reference, seeds = None, []
        for seed in (1, 2, 3):
            values = []
            for arm in (left, right):
                tag = 'b06' if arm == 'b' else f'{arm}500t{seed}'
                entry = reports[tag]['en_bpc'] if task == 'en_bpc' else reports[tag]['tasks'][language][task]
                if entry['status'] != 'complete':
                    problems.append({'tag': tag, 'task': task, **entry})
                    continue
                data = cached[tag]['en_bpc'] if task == 'en_bpc' else cached[tag][language][task]
                if reference is None:
                    reference = data['signature']
                if data['signature'] != reference:
                    problems.append({'tag': tag, 'task': task, 'status': 'invalid',
                                     'reason': 'Ordered item hashes, cohort, protocol or gold differ'})
                values.append(data['values'])
            if len(values) == 2 and values[0].shape == values[1].shape:
                seeds.append(values[0] - values[1])
        if len(seeds) == 3:
            differences.append(np.asarray(seeds))
    status = 'invalid' if any(p['status'] == 'invalid' for p in problems) else 'pending'
    result = {'status': status if problems else 'complete', 'problems': problems}
    if not problems:
        delta = np.mean([d.mean(axis=1) for d in differences], axis=0)
        result['seed_deltas'] = seed_summary({str(s + 1): float(v) for s, v in enumerate(delta)})
    return result, differences


def summarize(root: Path, cohort: dict, n_bootstrap: int = 10000, expected_bpc_docs: int = 1000) -> dict:
    counts = cohort['counts']
    if (set(counts) != set(TASKS) or any(set(counts[k]) != set(tasks) for k, tasks in TASKS.items())
            or any(type(n) is not int or n < 1 for group in counts.values() for n in group.values())):
        raise ValueError('Cohort counts must contain exactly the prespecified tasks and positive counts')
    if (set(cohort['protocols']) != set(TASKS) or set(cohort['tasks']) != set(TASKS)
            or not Path(cohort['base_model']['path']).is_absolute()
            or not digest_valid(cohort['base_model']['sha256'])
            or set(cohort['tokenizer_sha256']) != {'base', 'extended'}
            or not all(digest_valid(d) for d in cohort['tokenizer_sha256'].values())):
        raise ValueError('Frozen cohort identity is invalid')
    for lang, tasks in TASKS.items():
        if set(cohort['tasks'][lang]) != set(tasks) or not isinstance(cohort['protocols'][lang], dict) or not cohort['protocols'][lang]:
            raise ValueError('Frozen tasks or protocols are invalid')
        for task in tasks:
            frozen = cohort['tasks'][lang][task]
            if (frozen['n'] != counts[lang][task] or len(frozen['item_sha256']) != frozen['n']
                    or len(frozen['y_true']) != frozen['n']
                    or not all(digest_valid(h) for h in frozen['item_sha256'])
                    or frozen['cohort_sha256'] != cohort_digest(frozen['item_sha256'])):
                raise ValueError('Frozen cohort counts or hash are invalid')
    if n_bootstrap < 1 or expected_bpc_docs < 1:
        raise ValueError('Positive bootstrap and BPC document counts required')
    reports, cached = load_models(root, cohort, expected_bpc_docs)
    contrasts, arrays = {}, {}
    for endpoint in ('ar_accuracy', 'en_accuracy', 'en_bpc'):
        for name, (left, right) in PAIRS.items():
            key = f'{endpoint}:{name}'
            contrast, arrays[key] = paired_tasks(endpoint, left, right, reports, cached)
            contrasts[key] = {'endpoint': endpoint, 'left': left, 'right': right,
                              'units': 'bits per character' if endpoint == 'en_bpc' else 'percentage points',
                              **contrast}
    complete = all(c['status'] == 'complete' for c in contrasts.values())
    if complete:
        statistics = {k: (crossed_bootstrap(a[0], n_bootstrap) if k.startswith('en_bpc:')
                           else task_bootstrap(a, n_bootstrap)) for k, a in arrays.items()}
        corrected = holm({k: v['p_value'] for k, v in statistics.items()})
        for key, statistic in statistics.items():
            contrasts[key]['inference'] = {**statistic, **corrected[key], 'ci_scope': 'marginal'}
    groups = {}
    for arm, prefix in (('lep', 'l'), ('cpt', 'c')):
        group = {}
        for language in TASKS:
            group[language] = seed_summary({str(s): reports[f'{prefix}500t{s}']['macro_accuracy'][language]
                                           for s in (1, 2, 3) if language in reports[f'{prefix}500t{s}']['macro_accuracy']})
        group['en_bpc'] = seed_summary({str(s): reports[f'{prefix}500t{s}']['en_bpc']['mean']
                                       for s in (1, 2, 3) if reports[f'{prefix}500t{s}']['en_bpc']['status'] == 'complete'})
        groups[arm] = group
    return {'status': 'complete' if complete else 'pending', 'planned_contrasts': 9,
            'cohort_counts': counts, 'expected_bpc_docs': expected_bpc_docs, 'models': reports,
            'groups': groups, 'contrasts': contrasts, 'correction': 'Holm across all nine prespecified contrasts',
            'limitations': 'Three training seeds; approximate marginal bootstrap intervals. No equivalence claim.'}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('outputs/rebuttal'))
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path('outputs/rebuttal/downstream-v1/summary.json'))
    args = parser.parse_args()
    cohort = json.loads(args.cohort.read_text())
    result = summarize(args.root, cohort)
    result['cohort_receipt_sha256'] = hashlib.sha256(args.cohort.read_bytes()).hexdigest()
    result['implementation_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(f"Nine-test family: {result['status']}; wrote {args.out}")


if __name__ == '__main__':
    main()
