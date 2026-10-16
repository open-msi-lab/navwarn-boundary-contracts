"""Scoring aggregates and resampling statistics used for the content and processing results."""
import numpy as np

FIELDS = ('message_function', 'event_type', 'geometry', 'temporal',
          'restrictions', 'affected_entities', 'authority', 'lifecycle_refs')
CRITICAL = ('geometry', 'temporal', 'restrictions', 'affected_entities', 'lifecycle_refs')
MODES = ('S', 'N', 'NP')   # strict, normalized, normalized and part-blind matching
SEED = 20260913
N_BOOT = 10000


def pool(counts_by_field, fields=FIELDS):
    """Micro-pooled tp/fp/fn and F1 of one record over the given fields."""
    c = {k: sum(counts_by_field[f][k] for f in fields) for k in ('tp', 'fp', 'fn')}
    c['f1'] = 2 * c['tp'] / (2 * c['tp'] + c['fp'] + c['fn']) if 2 * c['tp'] + c['fp'] + c['fn'] else 0.
    c['exact'] = c['fp'] + c['fn'] == 0
    return c


def summary(evals):
    """Validity counts, micro F1, mean per-record F1 and exact-record counts of one response set."""
    result = {'n': len(evals), 'valid_n': sum(e['valid'] for e in evals.values()),
              'pipeline_field_valid_n': sum(e['pipeline_field_valid'] for e in evals.values())}
    for m in MODES:
        sums = {f: {k: sum(e['counts'][m][f][k] for e in evals.values()) for k in ('tp', 'fp', 'fn')}
                for f in FIELDS}
        vals = [pool(e['counts'][m]) for e in evals.values()]
        result[m] = {'micro': pool(sums), 'item_f1': float(np.mean([v['f1'] for v in vals])),
                     'exact_n': sum(v['exact'] for v in vals), 'fields': sums,
                     'critical_exact_n': sum(pool(e['counts'][m], CRITICAL)['exact'] for e in evals.values())}
    return result


def bootstrap_ci(values, n_boot=N_BOOT, alpha=0.05, seed=SEED):
    """Percentile bootstrap interval of the mean."""
    v = np.asarray(values, dtype=float)
    n = int(v.shape[0])
    point = float(np.mean(v))
    if n < 2:
        lo = hi = point
    else:
        rng = np.random.default_rng(seed)
        boot = np.empty(n_boot, dtype=float)
        for b in range(n_boot):
            boot[b] = np.mean(v[rng.integers(0, n, n)])
        lo, hi = (float(x) for x in np.percentile(boot, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)]))
    return {'point': point, 'lo': lo, 'hi': hi, 'n': n, 'n_boot': n_boot, 'alpha': alpha, 'seed': seed}


def paired(a, b, seed=SEED, alpha=0.05):
    """Paired percentile bootstrap interval of mean(a) - mean(b)."""
    av, bv = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert av.shape == bv.shape
    return bootstrap_ci(av - bv, alpha=alpha, seed=seed)


def permutation_p(a, b, seed=SEED):
    """Two-sided sign-flip test on paired differences; exact up to 18 nonzero differences."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.abs(d) > 1e-12]
    if not len(d):
        return {'p': 1., 'method': 'exact', 'nonzero': 0, 'draws': 1}
    target = abs(d.sum()) - 1e-12
    if len(d) <= 18:
        totals = np.array([0.])
        for x in d:
            totals = np.concatenate([totals + x, totals - x])
        return {'p': float(np.mean(np.abs(totals) >= target)), 'method': 'exact',
                'nonzero': len(d), 'draws': len(totals)}
    rng = np.random.default_rng(seed)
    v = rng.choice([-1., 1.], size=(10000, len(d))) @ d
    return {'p': float((1 + np.sum(np.abs(v) >= target)) / 10001), 'method': 'monte-carlo-plus-one',
            'nonzero': len(d), 'draws': 10000, 'seed': seed}


def holm_adjust(pvalues):
    """Holm step-down adjusted p-values, in input order."""
    p = [1.0 if (x is None or x != x) else float(x) for x in pvalues]
    m = len(p)
    adjusted, running = [0.0] * m, 0.0
    for rank, i in enumerate(sorted(range(m), key=lambda i: p[i])):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adjusted[i] = running
    return adjusted
