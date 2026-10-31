import ast
import itertools
import json
from pathlib import Path
import tempfile
import time
import tokenize
import io

import nbformat
import numpy as np
import pandas as pd
from nbclient import NotebookClient
from sklearn.datasets import make_classification

ROOT = Path(__file__).resolve().parents[1]
OUT = Path('/tmp/fds26-verification')
OUT.mkdir(exist_ok=True)

CHECKS = '''
import itertools, json
from pathlib import Path
parts = [df_fit, df_tune, df_cal, df_val]
assert all(set(a['id']).isdisjoint(b['id']) for a, b in itertools.combinations(parts, 2))
assert all(np.isfinite(X).all() and X.dtype == np.float32 for X in [X_fit, X_tune, X_cal, X_val])
assert np.array_equal(lr.classes_, knn.classes_)
assert np.array_equal(lr.classes_, np.arange(len(le.classes_)))
assert set(np.unique(correct_cal)) == {0, 1}
for p in [proba_val_lr, proba_val_knn]:
    np.testing.assert_allclose(p.sum(axis=1), 1)
assert confidence_score([0, 1], [0, 0], [1, 1]) == 1 / 6
assert confidence_score([0, 1], [0, 1], [1, 1]) == 1
assert confidence_score([0, 1], [0, 1], [0, 0]) == 0
try:
    confidence_score([0], [0], [np.nan])
except AssertionError:
    pass
else:
    raise AssertionError('NaN confidence accepted')
example = df_val.drop(columns='industry').iloc[[7, 2, 7, 1]].copy()
example.index = [44, 2, 44, -3]
original = example.copy(deep=True)
names, confidence = predict_industry(example)
assert names.shape == confidence.shape == (4,)
assert np.isfinite(confidence).all() and ((confidence >= 0) & (confidence <= 1)).all()
pd.testing.assert_frame_equal(example, original)
expected_indices = np.array([7, 2, 7, 1])
np.testing.assert_array_equal(names, le.inverse_transform(pred_val[expected_indices]))
np.testing.assert_allclose(confidence, confidence_methods['Financial-calibrated ensemble'][expected_indices])
with_target = example.assign(industry='INVALID TARGET')
np.testing.assert_array_equal(predict_industry(with_target)[0], names)
np.testing.assert_allclose(predict_industry(with_target)[1], confidence)
nan_finance = example.copy()
nan_finance[financial_columns] = np.nan
assert np.isfinite(predict_industry(nan_finance)[1]).all()
saved = {key: profiles[key].copy() for key in profiles}
for key in profiles:
    profiles[key] = np.ones(20)
ties = example.copy()
ties[financial_columns] = 1.0
p1 = np.zeros((len(ties), len(le.classes_)))
p1[:, 0] = 1
_, f = confidence_features(ties, p1, p1)
np.testing.assert_allclose(f[:, 3:], 1)
for key in profiles:
    profiles[key] = np.ones(19)
_, f = confidence_features(ties, p1, p1)
np.testing.assert_allclose(f[:, 3], 0.5)
np.testing.assert_allclose(f[:, 4], 0)
for key in profiles:
    profiles[key] = np.ones(20)
for value in [-1e100, 1e100]:
    tails = ties.copy()
    tails[financial_columns] = value
    _, f = confidence_features(tails, p1, p1)
    assert np.isfinite(f).all() and (f[:, 3] < 0.05).all()
profiles.clear()
profiles.update(saved)
snapshot = {'ids': [d['id'].tolist() for d in parts], 'weight': lr_weight,
            'features': features_val.tolist(), 'confidence': confidence_methods['Financial-calibrated ensemble'].tolist(),
            'predictions': pred_val.tolist(), 'lr_proba': proba_val_lr.tolist(), 'knn_proba': proba_val_knn.tolist(),
            'pca_variance': pca.explained_variance_ratio_.tolist()}
Path(SNAPSHOT_PATH).write_text(json.dumps(snapshot))
print('Split separation, real pipeline execution, scorer, alignment, inference order, no mutation, target independence, financial ranks and profile cutoff: PASS')
'''

NEW_CHECKS = '''
raw = example.drop(columns=financial_columns)
missing_names, missing_c = predict_industry(raw)
np.testing.assert_array_equal(missing_names, names)
np.testing.assert_allclose(missing_c, predict_industry(nan_finance)[1])
arrays = raw.copy()
arrays['business_description_embedding'] = arrays['business_description_embedding'].map(ast.literal_eval)
np.testing.assert_allclose(predict_industry(arrays)[1], missing_c)
inf_finance = example.copy()
inf_finance[financial_columns] = [np.inf, -np.inf, np.nan]
np.testing.assert_allclose(predict_industry(inf_finance)[1], missing_c)
finance = pd.DataFrame({'id': [example['id'].iloc[0], example['id'].iloc[0]],
                        'net_profit_margin': [2., 4.], 'ebitda_margin': [np.inf, 6.], 'asset_turnover': [-2., 2.]})
before = finance.copy(deep=True)
prepared = prepare_companies(raw, finance)
assert len(prepared) == len(raw)
assert prepared['id'].tolist() == raw['id'].tolist()
np.testing.assert_allclose(prepared.iloc[0][financial_columns].to_numpy(dtype=float), [3., 6., 0.])
pd.testing.assert_frame_equal(finance, before)
np.testing.assert_allclose(predict_industry(raw, finance)[1], predict_industry(prepared)[1])
np.testing.assert_allclose(predict_industry(example, finance)[1], predict_industry(prepared)[1])
partial_finance = finance[['id', 'net_profit_margin']]
partial_prepared = prepare_companies(raw, partial_finance)
assert partial_prepared[['ebitda_margin', 'asset_turnover']].isna().all().all()
assert np.isfinite(predict_industry(raw, partial_finance)[1]).all()
for column in financial_columns:
    partial = example.drop(columns=column)
    assert np.isfinite(predict_industry(partial)[1]).all()
print('Missing columns, infinities, list embeddings, separate duplicate-ID finances, override policy: PASS')
'''

def execute(path, work, label, extra):
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    for cell in nb.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    snapshot = str(OUT / (label + '.json'))
    nb.cells.append(nbformat.v4.new_code_cell('SNAPSHOT_PATH = ' + repr(snapshot) + '\n' + CHECKS + extra))
    begin = time.perf_counter()
    NotebookClient(nb, timeout=180, kernel_name='python3', resources={'metadata': {'path': str(work)}}).execute()
    nbformat.write(nb, OUT / (label + '-synthetic.ipynb'))
    print(label, 'PASS', round(time.perf_counter()-begin, 2), 'seconds; executed code cells:',
          len([c for c in nb.cells if c.cell_type == 'code']), flush=True)
    return json.loads(Path(snapshot).read_text())

def main():
    nb = nbformat.read(ROOT / 'industry_classification.ipynb', as_version=4)
    study = nbformat.read(ROOT / 'industry_classification_commented.ipynb', as_version=4)
    nbformat.validate(study)
    clean_code = [c.source for c in nb.cells if c.cell_type == 'code']
    study_code = [c.source for c in study.cells if c.cell_type == 'code']
    assert [ast.dump(ast.parse(s)) for s in clean_code] == [ast.dump(ast.parse(s)) for s in study_code]
    skeleton = nbformat.read(ROOT / 'archive/code_skeleton_industry_classification.ipynb', as_version=4)
    expected_tasks = [c.source for c in skeleton.cells if c.cell_type == 'markdown']
    actual_tasks = iter(c.source for c in nb.cells if c.cell_type == 'markdown')
    assert all(any(text == expected for text in actual_tasks) for expected in expected_tasks)
    assert skeleton.cells[19].source in clean_code
    assert skeleton.cells[20].source in clean_code
    for document in [nb, study]:
        code_cells = [c for c in document.cells if c.cell_type == 'code']
        if any(c.execution_count is not None for c in code_cells):
            assert all(c.execution_count is not None for c in code_cells)
            assert not any(o.output_type == 'error' for c in code_cells for o in c.outputs)
    for cell in nb.cells:
        if cell.cell_type == 'code':
            ast.parse(cell.source)
            if cell.source != skeleton.cells[20].source:
                assert not any(t.type == tokenize.COMMENT for t in tokenize.generate_tokens(io.StringIO(cell.source).readline))
    with tempfile.TemporaryDirectory(prefix='fds26-synthetic-') as work:
        work = Path(work)
        (work / 'data').mkdir()
        X, y = make_classification(n_samples=1800, n_features=768, n_informative=24, n_redundant=8,
                                   n_classes=8, n_clusters_per_class=2, class_sep=0.9, flip_y=0.15, random_state=7)
        rng = np.random.default_rng(42)
        df = pd.DataFrame({'id': np.arange(len(X)), 'industry': ['Industry ' + str(i) for i in y],
                           'business_description_embedding': [repr(v.round(4).tolist()) for v in X]})
        df = pd.concat([df, df.iloc[[3]].assign(business_description_embedding=df['business_description_embedding'].iloc[4])], ignore_index=True)
        fin = pd.DataFrame({'id': np.arange(len(X)), 'net_profit_margin': rng.normal(0, 0.3, len(X)),
                            'ebitda_margin': rng.normal(0, 0.5, len(X)), 'asset_turnover': rng.lognormal(0, 1, len(X))})
        fin.loc[::7, 'ebitda_margin'] = np.nan
        fin.loc[::31, 'net_profit_margin'] = np.inf
        fin.loc[::67, 'asset_turnover'] = -np.inf
        fin = pd.concat([fin.iloc[10:], fin.iloc[[11, 12, 13]]], ignore_index=True)
        df.to_pickle(work / 'data/df_train.pkl')
        fin.to_pickle(work / 'data/df_financials_train.pkl')
        old = execute(ROOT / 'archive/industry_classification_baseline.ipynb', work, 'baseline', '')
        new = execute(ROOT / 'industry_classification.ipynb', work, 'final', NEW_CHECKS)
        assert old['ids'] == new['ids'] and old['weight'] == new['weight']
        np.testing.assert_array_equal(old['predictions'], new['predictions'])
        for key in ['features', 'confidence', 'lr_proba', 'knn_proba', 'pca_variance']:
            np.testing.assert_allclose(old[key], new[key], atol=1e-10, rtol=1e-10)
        print('Baseline/final equivalence on synthetic fixture: PASS (atol=rtol=1e-10)', flush=True)
        report = {'data': 'synthetic only; NOT assignment performance', 'rows': len(df), 'dimensions': 768,
                  'classes': 8, 'split_rows': [len(p) for p in new['ids']], 'checks': 'PASS',
                  'baseline_final_equal': True, 'tolerance': 1e-10}
        (OUT / 'report.json').write_text(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
