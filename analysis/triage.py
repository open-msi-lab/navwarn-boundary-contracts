"""Inspection allocation: grouped cross-fitted review scores and residual error by quota.

Reads data/observable-records.json.gz and writes data/triage-results.json,
data/triage-models.json and data/triage-scores.json.gz.

    python analysis/triage.py
"""
from pathlib import Path
from collections import defaultdict
import gzip, hashlib, json
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
QUOTAS = [.1, .2, .3, .4, .5]
CORE = ['geometry', 'temporal', 'lifecycle_refs']
FIVE = CORE + ['restrictions', 'affected_entities']
METHODS = ['uniform', 'incident', 'content_first', 'incident_first', 'text_length',
           'structural_learner', 'full_learner', 'oracle']

def dump(name, x):
    (DATA / name).write_text(json.dumps(x, sort_keys=True, indent=2) + '\n')

def objective(theta, x, y):
    z = theta[0] + x @ theta[1:]
    loss = np.logaddexp(0, z).sum() - y @ z + .5 * (theta[1:] @ theta[1:])
    residual = expit(z) - y
    return loss, np.r_[residual.sum(), x.T @ residual + theta[1:]]

def fit(x, y):
    mean = x.mean(axis=0); scale = x.std(axis=0); scale[scale < 1e-12] = 1
    z = (x - mean) / scale
    out = minimize(objective, np.zeros(x.shape[1]+1), args=(z, y), jac=True,
                   method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
    assert out.success, out.message
    return {'mean': mean.tolist(), 'scale': scale.tolist(), 'theta': out.x.tolist(),
            'training_n': len(y), 'training_errors': int(y.sum()),
            'gradient_max': float(np.max(np.abs(out.jac)))}

def predict(model, x):
    z = (x - np.array(model['mean'])) / np.array(model['scale'])
    theta = np.array(model['theta'])
    return expit(theta[0] + z @ theta[1:])

def errors(rows, fields=CORE, mode='N'):
    return np.array([any(r['counts'][mode][f]['fp'] + r['counts'][mode][f]['fn'] for f in fields) for r in rows], float)

def split_groups(rows, seed):
    groups = defaultdict(set)
    for r in rows:
        groups[r['template_sha256']].add(r['uid'])
    ordered = sorted(groups, key=lambda k: (-len(groups[k]), hashlib.sha256(f'{seed}:{k}'.encode()).hexdigest()))
    size = [0] * 5; assignments = {}
    for key in ordered:
        fold = min(range(5), key=lambda f: (size[f], f))
        assignments[key] = fold; size[fold] += len(groups[key])
    return np.array([assignments[r['template_sha256']] for r in rows]), assignments, size

def retention(scores, y, q):
    """Vectorized exact uniform allocation at the boundary score tie."""
    scores = np.atleast_2d(np.asarray(scores, float)); y = np.atleast_2d(y)
    n = scores.shape[1]; k = round(n * q)
    assert 0 < k < n
    cutoff = np.partition(scores, n-k, axis=1)[:, n-k, None]
    above = scores > cutoff; tied = scores == cutoff
    remain = k - above.sum(axis=1)
    share = remain / tied.sum(axis=1)
    caught = (above*y).sum(axis=1) + share*(tied*y).sum(axis=1)
    residual = (y.sum(axis=1)-caught)/(n-k)
    return residual, caught

def description(scores, y, q):
    risk, caught = retention(scores, y, q); n = len(y); k = round(n*q)
    return {'n': n, 'quota_n': k, 'reference_error_n': int(y.sum()),
            'selected_errors': float(caught[0]), 'selected_reference_exact': float(k-caught[0]),
            'error_recall': float(caught[0]/y.sum()) if y.sum() else None,
            'uninspected_errors': float(y.sum()-caught[0]), 'residual': float(risk[0])}

def learn_scores(rows, matrix, labels, seed):
    report = np.array([r['set']=='Reporting' for r in rows]); val = ~report
    report_rows = [r for r in rows if r['set']=='Reporting']
    folds, assignment, sizes = split_groups(report_rows, seed)
    rep_idx = np.where(report)[0]; val_idx = np.where(val)[0]
    score = np.full(len(rows), np.nan); fits = []; checks = []
    for fold in range(5):
        tr = rep_idx[folds != fold]; te = rep_idx[folds == fold]
        for attr in ('uid','text_sha256','template_sha256'):
            assert not {rows[i][attr] for i in tr} & {rows[i][attr] for i in te}
        model = fit(matrix[tr], labels[tr]); score[te] = predict(model, matrix[te])
        fits.append({'fold': fold, **model})
        checks.append({'fold': fold, 'train_n': len(tr), 'test_n': len(te), 'group_overlap': 0})
    val_templates = {rows[i]['template_sha256'] for i in val_idx}
    tr = np.array([i for i in rep_idx if rows[i]['template_sha256'] not in val_templates])
    for attr in ('uid','text_sha256','template_sha256'):
        assert not {rows[i][attr] for i in tr} & {rows[i][attr] for i in val_idx}
    model = fit(matrix[tr], labels[tr]); score[val_idx] = predict(model, matrix[val_idx])
    assert np.isfinite(score).all()
    return score, {'fold_fits': fits, 'transfer_fit': model, 'fold_warning_sizes': sizes,
                   'template_assignment': assignment, 'fold_checks': checks,
                   'transfer_excluded_reporting_rows': int(report.sum()-len(tr)),
                   'transfer_training_indices': tr.tolist(), 'transfer_validation_indices': val_idx.tolist()}

def main():
    source = DATA/'observable-records.json.gz'
    data = json.loads(gzip.decompress(source.read_bytes())); rows = data['rows']
    x = np.array([r['features'] for r in rows]); names = data['feature_names']
    # Sol is the reference category; three remaining model indicators.
    onehot = np.array([[int(r['model']==m) for m in data['model_names'][1:]] for r in rows])
    xx = np.column_stack([x, onehot]); all_names = names + data['model_names'][1:]
    structural = [i for i,n in enumerate(all_names) if n not in ('geometry_flag','temporal_flag')]
    y = errors(rows); base_score = {
        'uniform': np.zeros(len(rows)), 'incident': x[:,7],
        'content_first': np.array([r['checker_score'] for r in rows], float),
        'incident_first': 3*x[:,7] + x[:,9] + x[:,10], 'text_length': x[:,0], 'oracle': y}
    fits = {}; sensitivity = defaultdict(dict)
    for name, cols in [('structural_learner', structural), ('full_learner', list(range(xx.shape[1])))]:
        for seed in range(20260914,20260919):
            scores, meta = learn_scores(rows, xx[:,cols], y, seed)
            if seed == 20260914:
                base_score[name] = scores; fits[name] = {'feature_names': [all_names[i] for i in cols], **meta}
            for config in sorted({r['configuration'] for r in rows if r['set']=='Reporting'}):
                ii = [i for i,r in enumerate(rows) if r['configuration']==config]
                sensitivity[name].setdefault(config, {})[str(seed)] = description(scores[ii],y[ii],.2)['residual']
    results = {}; contrasts = {}; rng = np.random.default_rng(20260914)
    for config in sorted({r['configuration'] for r in rows}):
        ii = np.array([i for i,r in enumerate(rows) if r['configuration']==config]); rr = [rows[i] for i in ii]
        results[config] = {}
        for group, fields in [('core', CORE), ('five', FIVE)]:
            yy = errors(rr, fields)
            results[config][group] = {method: {f'{q:.1f}': description((yy if method=='oracle' else base_score[method][ii]), yy,q)
                                               for q in QUOTAS} for method in METHODS}
        strata = {}
        for frame in sorted({r['source_frame'] for r in rr}):
            jj = np.array([i for i in ii if rows[i]['source_frame']==frame])
            strata[frame] = {m: description(base_score[m][jj], y[jj], .2) for m in METHODS}
        results[config]['source_frame_core_20'] = strata
        boot = np.concatenate([rng.choice([j for j,r in enumerate(rr) if r['source_frame']==frame],
                    size=(10000, sum(r['source_frame']==frame for r in rr)))
                    for frame in sorted({r['source_frame'] for r in rr})], axis=1)
        yy = y[ii]; a = base_score['full_learner'][ii]; b = base_score['incident'][ii]
        delta = retention(a[boot], yy[boot], .2)[0] - retention(b[boot], yy[boot], .2)[0]
        point = retention(a, yy,.2)[0][0] - retention(b,yy,.2)[0][0]
        contrasts[config] = {'point': float(point), 'ci95': np.quantile(delta,[.025,.975]).tolist(),
                             'family_ci99': np.quantile(delta,[.005,.995]).tolist(),
                             'resamples': 10000, 'conditional_on_fitted_scores': True}
    score_rows = [{k:r[k] for k in ('configuration','uid','set','source_frame','template_sha256')} |
                  {'core_error':int(y[i]),'scores':{m:float(base_score[m][i]) for m in METHODS}}
                  for i,r in enumerate(rows)]
    raw=(json.dumps(score_rows,sort_keys=True,separators=(',',':'))+'\n').encode()
    (DATA/'triage-scores.json.gz').write_bytes(gzip.compress(raw,mtime=0))
    dump('triage-models.json',fits)
    dump('triage-results.json',{'design':'Retrospective grouped cross-fitting and transfer',
          'seed':20260914,'quotas':QUOTAS,'configurations':results,'primary_contrasts':contrasts,
          'fold_sensitivity':sensitivity,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
          'new_model_calls':0,'new_annotations':0})
    print(json.dumps({'20pct_core':{c:{m:round(v['core'][m]['0.2']['residual']*100,3) for m in METHODS}
                                   for c,v in results.items()},'contrasts':contrasts},indent=2))

if __name__ == '__main__':
    main()
