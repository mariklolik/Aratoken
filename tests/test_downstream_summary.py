import hashlib
import importlib.util
import json

import numpy as np
import pytest


def module():
    assert importlib.util.find_spec('scripts.eval.downstream_summary') is not None, 'seed-aware summary is absent'
    from scripts.eval import downstream_summary
    return downstream_summary


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def panel(root, populated=True):
    tasks = {'ar': {'ArabicMMLU': 2, 'MadinahQA': 3, 'ArabicEXAMS': 4, 'AraTrust': 5, 'AlGhafa': 6},
             'en': {'MMLU': 2, 'ARC-Easy': 6}}
    for tag in ['b06'] + [f'{arm}500t{s}' for arm in ('l', 'c') for s in (1, 2, 3)]:
        run = f'budget500m_{"hybrid" if tag[0] == "l" else "cpt"}_s{tag[-1]}'
        model = str(root / 'snapshots/base06') if tag == 'b06' else str(root / 'runs' / run / 'model/final_model')
        for lang, counts in tasks.items():
            directory = root / 'downstream-v1/mcq' / ('en' if lang == 'en' else '')
            for task, n in counts.items():
                correct = n if tag.startswith('l') else 0
                write_json(directory / f'{tag}_{task}.json', {
                    'task': task, 'model': model, 'model_sha256': hashlib.sha256(model.encode()).hexdigest(),
                    'tokenizer_sha256': ('e' if tag.startswith('l') else 'b') * 64,
                    'n': n, 'y_true': [0] * n,
                    'y_pred': [0] * correct + [1] * (n - correct),
                    'item_sha256': [hashlib.sha256(f'{task}:{i}'.encode()).hexdigest() for i in range(n)],
                    'cohort_sha256': hashlib.sha256(json.dumps(
                        [hashlib.sha256(f'{task}:{i}'.encode()).hexdigest() for i in range(n)],
                        ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
                    'protocol': {'name': 'mcq-letter-v2', 'lang': lang}})
        name = 'base' if tag == 'b06' else run
        path = root / 'base06_common_en.json' if tag == 'b06' else root / 'runs' / run / 'bpc_en.json'
        value = 1. if tag.startswith('l') else 2.
        write_json(path, {'n_docs': 3, 'results': {name: {
            'documents': [{'index': i, 'source_sha256': f'source{i}', 'text_sha256': f'text{i}',
                           'chars': 512, 'bpc': value} for i in range(3)],
            'protocol': {'name': 'common-text-bpc-v1', 'common_text_chars': 512}}}})
    frozen = {'counts': tasks, 'protocols': {}, 'tasks': {},
              'base_model': {'path': str(root / 'snapshots/base06'),
                             'sha256': hashlib.sha256(str(root / 'snapshots/base06').encode()).hexdigest()},
              'tokenizer_sha256': {'base': 'b' * 64, 'extended': 'e' * 64}}
    for lang, counts in tasks.items():
        frozen['tasks'][lang] = {}
        for task in counts:
            path = root / 'downstream-v1/mcq' / ('en' if lang == 'en' else '') / f'b06_{task}.json'
            data = json.loads(path.read_text())
            frozen['protocols'][lang] = data['protocol']
            frozen['tasks'][lang][task] = {k: data[k] for k in ('n', 'item_sha256', 'cohort_sha256', 'y_true')}
    if not populated:
        for path in root.rglob('*.json'):
            path.unlink()
    return frozen


def test_empty_panel_retains_all_nine_contrasts_and_seven_models(tmp_path):
    m = module()
    counts = panel(tmp_path, populated=False)
    r = m.summarize(tmp_path, counts, n_bootstrap=19, expected_bpc_docs=3)
    assert len(r['models']) == 7
    assert len(r['contrasts']) == 9
    assert r['status'] == 'pending'
    assert all('inference' not in c for c in r['contrasts'].values())


def test_complete_family_has_units_seed_values_and_nine_test_correction(tmp_path):
    m = module()
    r = m.summarize(tmp_path, panel(tmp_path), n_bootstrap=19, expected_bpc_docs=3)
    assert r['status'] == 'complete'
    for endpoint in ('ar_accuracy', 'en_accuracy', 'en_bpc'):
        c = r['contrasts'][f'{endpoint}:lep-cpt']
        delta = -1 if endpoint == 'en_bpc' else 100
        assert c['seed_deltas']['values'] == {'1': delta, '2': delta, '3': delta}
        assert c['inference']['mean_diff'] == delta
        assert c['inference']['p_raw'] == pytest.approx(.05)
        assert c['inference']['p_holm'] == pytest.approx(.45)
        assert c['inference']['ci_scope'] == 'marginal'
    assert r['models']['b06']['tasks']['ar']['ArabicMMLU']['accuracy'] == 0
    assert r['models']['l500t1']['macro_accuracy']['ar'] == 100


@pytest.mark.parametrize('field,value', [
    ('item_sha256', ['a' * 64, 'b' * 64]), ('cohort_sha256', 'b' * 64),
    ('protocol', {'version': 'wrong'}), ('y_true', [1, 0]), ('n', 99),
    ('y_pred', [0]), ('model', '/wrong/model')])
def test_invalid_receipt_withholds_entire_family(tmp_path, field, value):
    m = module()
    counts = panel(tmp_path)
    path = tmp_path / 'downstream-v1/mcq/l500t3_ArabicMMLU.json'
    data = json.loads(path.read_text())
    data[field] = value
    path.write_text(json.dumps(data))
    r = m.summarize(tmp_path, counts, n_bootstrap=19, expected_bpc_docs=3)
    assert r['status'] == 'pending'
    assert any(c['status'] == 'invalid' for c in r['contrasts'].values())
    assert all('inference' not in c for c in r['contrasts'].values())


def test_missing_seed_never_drops_family_denominator(tmp_path):
    m = module()
    counts = panel(tmp_path)
    (tmp_path / 'downstream-v1/mcq/en/c500t3_MMLU.json').unlink()
    r = m.summarize(tmp_path, counts, n_bootstrap=19, expected_bpc_docs=3)
    assert r['status'] == 'pending'
    assert len(r['contrasts']) == 9
    assert all('inference' not in c for c in r['contrasts'].values())
    assert r['models']['c500t3']['tasks']['en']['ARC-Easy']['accuracy'] == 0


def test_macro_is_unweighted_and_base_is_fixed_reference(tmp_path):
    m = module()
    counts = panel(tmp_path)
    path = tmp_path / 'downstream-v1/mcq/en/l500t1_ARC-Easy.json'
    data = json.loads(path.read_text())
    data['y_pred'] = [1] * 6
    path.write_text(json.dumps(data))
    r = m.summarize(tmp_path, counts, n_bootstrap=99, expected_bpc_docs=3)
    assert r['models']['l500t1']['macro_accuracy']['en'] == 50
    assert r['models']['l500t1']['pooled_accuracy']['en'] == 25
    assert r['contrasts']['en_accuracy:lep-base']['seed_deltas']['values'] == {'1': 50, '2': 100, '3': 100}


def test_task_bootstrap_shares_seed_draw_and_keeps_equal_task_weight():
    m = module()
    a = np.array([[-1.] * 2, [0.] * 2, [1.] * 2])
    b = np.repeat(-a[:, :1], 7, axis=1)
    result = m.task_bootstrap([a, b], 199, 42)
    assert result['mean_diff'] == result['ci_lower'] == result['ci_upper'] == 0
    assert result['p_value'] == 1
    varied = m.task_bootstrap([np.tile([-1., 1.], (3, 1)), np.zeros((3, 7))], 199, 42)
    assert varied['ci_lower'] < 0 < varied['ci_upper']
    assert varied == m.task_bootstrap([np.tile([-1., 1.], (3, 1)), np.zeros((3, 7))], 199, 42)


def test_bpc_hash_mismatch_blocks_family(tmp_path):
    m = module()
    counts = panel(tmp_path)
    path = tmp_path / 'runs/budget500m_cpt_s3/bpc_en.json'
    data = json.loads(path.read_text())
    data['results']['budget500m_cpt_s3']['documents'][0]['text_sha256'] = 'different'
    path.write_text(json.dumps(data))
    r = m.summarize(tmp_path, counts, n_bootstrap=19, expected_bpc_docs=3)
    assert r['contrasts']['en_bpc:lep-cpt']['status'] == 'invalid'
    assert all('inference' not in c for c in r['contrasts'].values())


def test_counts_must_cover_exact_prespecified_tasks(tmp_path):
    m = module()
    counts = panel(tmp_path)
    del counts['counts']['ar']['AlGhafa']
    with pytest.raises(ValueError, match='counts'):
        m.summarize(tmp_path, counts, n_bootstrap=19)


def test_shared_corrupt_cohort_cannot_pass_by_agreement(tmp_path):
    m = module()
    counts = panel(tmp_path)
    for path in (tmp_path / 'downstream-v1/mcq').glob('*_ArabicMMLU.json'):
        data = json.loads(path.read_text())
        data['cohort_sha256'] = 'f' * 64
        path.write_text(json.dumps(data))
    r = m.summarize(tmp_path, counts, n_bootstrap=19, expected_bpc_docs=3)
    assert r['status'] == 'pending'
    assert r['contrasts']['ar_accuracy:lep-cpt']['status'] == 'invalid'
    assert all('inference' not in c for c in r['contrasts'].values())


@pytest.mark.parametrize('field', ['protocol', 'y_true', 'item_sha256'])
def test_mutually_consistent_receipts_must_match_frozen_preflight(tmp_path, field):
    m = module()
    frozen = panel(tmp_path)
    for path in (tmp_path / 'downstream-v1/mcq').glob('*_ArabicMMLU.json'):
        data = json.loads(path.read_text())
        if field == 'protocol':
            data[field]['rotation'] = 'changed'
        elif field == 'y_true':
            data[field] = [1, 1]
        else:
            data[field] = ['a' * 64, 'b' * 64]
            data['cohort_sha256'] = hashlib.sha256(json.dumps(data[field], separators=(',', ':')).encode()).hexdigest()
        path.write_text(json.dumps(data))
    result = m.summarize(tmp_path, frozen, n_bootstrap=19, expected_bpc_docs=3)
    assert result['status'] == 'pending'
    assert result['contrasts']['ar_accuracy:lep-cpt']['status'] == 'invalid'
    assert all('inference' not in c for c in result['contrasts'].values())


@pytest.mark.parametrize('tag,field,value', [
    ('b06', 'model', '/other/base'), ('b06', 'model_sha256', 'a' * 64),
    ('c500t1', 'tokenizer_sha256', 'e' * 64), ('l500t1', 'tokenizer_sha256', 'b' * 64),
    ('l500t1', 'model_sha256', 'a' * 64), ('c500t2', 'model_sha256', 'not-a-digest')])
def test_model_and_tokenizer_identity_cannot_be_substituted(tmp_path, tag, field, value):
    m = module()
    frozen = panel(tmp_path)
    path = tmp_path / f'downstream-v1/mcq/en/{tag}_MMLU.json'
    data = json.loads(path.read_text())
    data[field] = value
    path.write_text(json.dumps(data))
    result = m.summarize(tmp_path, frozen, n_bootstrap=19, expected_bpc_docs=3)
    assert result['status'] == 'pending'
    assert all('inference' not in c for c in result['contrasts'].values())
    assert result['models'][tag]['tasks']['en']['MMLU']['status'] == 'invalid'
