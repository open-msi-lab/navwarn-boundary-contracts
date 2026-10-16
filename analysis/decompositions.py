"""Descriptive decompositions of the inspection and exchange results (Tables 8 and 10, Figs 4 and 6).

Inspection: natural-error recall, false-alarm rate, precision and positive
likelihood ratio of the two binary review alerts (processing incident; any
content-check flag), the composition of natural core errors, injected-corruption
coverage, and the tie-expected allocation identity R(q) = (|E| - S(k)) / (n - k)
including Corollary 1. Every recomputed allocation is asserted against
data/triage-results.json.

Exchange: detectability of a latitude/longitude transposition by coordinate
range checks, by source frame. A transposed WGS 84 position stays within legal
ranges iff |longitude| <= 90 degrees. Totals are asserted against
data/geospatial-results.json.

    python analysis/decompositions.py

Writes data/decompositions.json (aggregate counts only).
"""
from collections import Counter, defaultdict
from pathlib import Path
import gzip
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OBSERVABLE = DATA / 'observable-records.json.gz'
TRIAGE = DATA / 'triage-results.json'
GEO = DATA / 'geospatial-results.json'
POSITIONS = DATA / 'geometry-positions.json.gz'
MUTATION = DATA / 'fault-injection.json'
OUT = DATA / 'decompositions.json'
BIN_DEG = 5

CORE = ('geometry', 'temporal', 'lifecycle_refs')
QUOTA = 0.2
TOL = 1e-9


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_selected(records, score, k):
    """Expected selected errors with uniform selection inside score ties."""
    tiers = defaultdict(list)
    for r in records:
        tiers[score(r)].append(r)
    remaining, selected = k, 0.0
    for s in sorted(tiers, reverse=True):
        if remaining <= 0:
            break
        tier = tiers[s]
        take = min(remaining, len(tier))
        selected += take / len(tier) * sum(r['error'] for r in tier)
        remaining -= take
    return selected


def alert_summary(records, alert):
    n = len(records)
    errors = sum(r['error'] for r in records)
    alerted = [r for r in records if alert(r)]
    hits = sum(r['error'] for r in alerted)
    false_alarms = len(alerted) - hits
    recall = hits / errors if errors else None
    false_alarm_rate = false_alarms / (n - errors) if n > errors else None
    if recall is None or false_alarm_rate is None:
        lr = None
    elif false_alarm_rate == 0:
        lr = 'inf' if recall > 0 else None
    else:
        lr = recall / false_alarm_rate
    return {'alerted': len(alerted), 'alerted_with_error': hits,
            'natural_recall': recall, 'false_alarm_rate': false_alarm_rate,
            'precision': hits / len(alerted) if alerted else None,
            'positive_likelihood_ratio': lr}


def lr_exceeds_one(summary):
    lr = summary['positive_likelihood_ratio']
    return lr == 'inf' or (lr is not None and lr > 1)


def inspection():
    data = json.loads(gzip.decompress(OBSERVABLE.read_bytes()))
    names = data['feature_names']
    triage = json.loads(TRIAGE.read_text())
    frozen = triage['configurations']
    quotas = triage['quotas']
    sets = defaultdict(list)
    for row in data['rows']:
        counts = row['counts']['N']
        feats = dict(zip(names, row['features']))
        wrong = {f: counts[f]['fp'] > 0 or counts[f]['fn'] > 0 for f in CORE}
        error = any(wrong.values())
        flags = int(feats['geometry_flag'] > 0) + int(feats['temporal_flag'] > 0)
        sets[row['configuration']].append({
            'error': int(error),
            'incident': int(bool(row['incident'])),
            'content_flags': flags,
            # A value-membership check can only flag an error that contains a
            # predicted geometry or temporal value absent from the reference.
            'checkable': int(error and (counts['geometry']['fp'] > 0 or counts['temporal']['fp'] > 0)),
            'omission_only': int(error and all(counts[f]['fp'] == 0 for f in CORE)),
            'lifecycle_only': int(error and not wrong['geometry'] and not wrong['temporal']),
            # The two groups above overlap (a missing lifecycle reference is both);
            # the overlap is counted so that the composition can be stated exactly.
            'omission_and_lifecycle': int(error and all(counts[f]['fp'] == 0 for f in CORE)
                                          and not wrong['geometry'] and not wrong['temporal']),
        })
    results, pooled = {}, defaultdict(Counter)
    policies = {
        'incident': lambda r: r['incident'],
        'content_first': lambda r: 2 * r['content_flags'] + r['incident'],
    }
    agreement = []
    for key, records in sorted(sets.items()):
        n = len(records)
        k = round(QUOTA * n)
        errors = sum(r['error'] for r in records)
        prevalence = errors / n
        alerts = {
            'incident': alert_summary(records, lambda r: r['incident'] == 1),
            'any_content_flag': alert_summary(records, lambda r: r['content_flags'] > 0),
        }
        frozen_core = frozen[key]['core']
        allocation = {}
        for name, score in policies.items():
            selected = expected_selected(records, score, k)
            residual = (errors - selected) / (n - k)
            ref = frozen_core[name][str(QUOTA)]
            assert abs(selected - ref['selected_errors']) < TOL, (key, name, selected, ref)
            assert abs(residual - ref['residual']) < TOL, (key, name, residual, ref)
            allocation[name] = {'selected_errors': selected, 'residual': residual}
        uniform = frozen_core['uniform'][str(QUOTA)]['residual']
        assert abs(uniform - prevalence) < TOL
        allocation['uniform'] = {'selected_errors': k * prevalence, 'residual': uniform}
        for policy, alert in (('incident', 'incident'), ('content_first', 'any_content_flag')):
            predicted = lr_exceeds_one(alerts[alert])
            observed = allocation[policy]['residual'] < uniform
            agreement.append({'set': key, 'policy': policy,
                              'predicted_better_than_random': predicted,
                              'observed_better_than_random': observed})
        composition = {name: sum(r[name] for r in records)
                       for name in ('checkable', 'omission_only', 'lifecycle_only',
                                    'omission_and_lifecycle')}
        composition['not_checkable'] = errors - composition['checkable']
        # Not-checkable errors that are neither pure omissions nor lifecycle-only
        # (an omission in geometry or time combined with a lifecycle error).
        composition['other_not_checkable'] = (composition['not_checkable'] - composition['omission_only']
                                              - composition['lifecycle_only']
                                              + composition['omission_and_lifecycle'])
        assert composition['other_not_checkable'] >= 0, (key, composition)
        # Saturation identity (manuscript Corollary 1): for the binary incident alert
        # used with alert-first priority and uniform fill, the residual error equals
        # p - m_k (p_a - p) / (n - m_k) with m_k = min(m, k), at every quota. Asserted
        # against every frozen incident-only residual.
        inc = alerts['incident']
        saturation = {}
        for q in quotas:
            kq = round(q * n)
            m_k = min(inc['alerted'], kq)
            predicted = prevalence - m_k * (inc['precision'] - prevalence) / (n - m_k)
            observed = frozen_core['incident'][str(q)]['residual']
            assert abs(predicted - observed) < TOL, (key, q, predicted, observed)
            saturation[str(q)] = {'quota_k': kq, 'alerted_m': inc['alerted'],
                                  'predicted_residual': predicted, 'frozen_residual': observed}
        results[key] = {'n': n, 'quota_k': k, 'core_errors': errors, 'prevalence': prevalence,
                        'alerts': alerts, 'allocation_at_20pct': allocation,
                        'error_composition': composition,
                        'saturation_identity_incident_only': saturation}
        if key != 'Sol/validation':
            P = pooled['reporting']
            P['n'] += n
            P['errors'] += errors
            for alert in ('incident', 'any_content_flag'):
                P[alert + '_alerted'] += alerts[alert]['alerted']
                P[alert + '_hits'] += alerts[alert]['alerted_with_error']
            for name, value in composition.items():
                P[name] += value
    P = pooled['reporting']
    pooled_summary = {'n': P['n'], 'core_errors': P['errors'],
                      'error_composition': {name: P[name] for name in
                                            ('checkable', 'not_checkable', 'omission_only', 'lifecycle_only',
                                             'omission_and_lifecycle', 'other_not_checkable')}}
    assert (P['omission_only'] + P['lifecycle_only'] - P['omission_and_lifecycle']
            + P['other_not_checkable'] == P['not_checkable'])
    for alert in ('incident', 'any_content_flag'):
        alerted, hits = P[alert + '_alerted'], P[alert + '_hits']
        recall = hits / P['errors']
        far = (alerted - hits) / (P['n'] - P['errors'])
        pooled_summary[alert] = {'alerted': alerted, 'alerted_with_error': hits,
                                 'natural_recall': recall, 'false_alarm_rate': far,
                                 'precision': hits / alerted, 'positive_likelihood_ratio': recall / far}
    assert P['n'] == 600 and P['errors'] == 75
    injected = {}
    for checker, detection in json.loads(MUTATION.read_text()).items():
        injected[checker] = {field: {'cases': v['n'], 'mutated_field_alerts': v['mutated_field_flagged'],
                                     'newly_triggered_alerts': v['new_field_detection'],
                                     'new_alert_coverage': v['new_field_detection'] / v['n']}
                             for field, v in detection.items()}
    assert injected['grammar_corrected_checker']['geometry']['newly_triggered_alerts'] == 55
    assert injected['grammar_corrected_checker']['temporal']['newly_triggered_alerts'] == 30
    saturation_points = sum(len(v['saturation_identity_incident_only']) for v in results.values())
    saturation_dev = max(abs(pt['predicted_residual'] - pt['frozen_residual'])
                         for v in results.values()
                         for pt in v['saturation_identity_incident_only'].values())
    return {'per_set': results, 'pooled_reporting': pooled_summary,
            'injected_corruption_coverage': injected,
            'criterion_agreement': agreement,
            'criterion_agreement_count': sum(a['predicted_better_than_random'] ==
                                             a['observed_better_than_random'] for a in agreement),
            'criterion_comparisons': len(agreement),
            'saturation_identity_points_checked': saturation_points,
            'saturation_identity_max_abs_deviation': saturation_dev}


def transposition():
    derived = json.loads(gzip.decompress(POSITIONS.read_bytes()))
    frozen = json.loads(GEO.read_text())
    counts = defaultdict(Counter)
    histogram = defaultdict(Counter)
    for key in sorted(derived):
        rejected = {r['uid'] for r in frozen['configurations'][key]['rejections']}
        for uid, record in sorted(derived[key]['projected'].items()):
            if uid in rejected:
                continue
            frame = 'china_msa' if uid.startswith('zh:') else 'navarea_iv'
            for geometry in record.get('geometry') or []:
                for lat, lon in geometry['coordinates']:
                    lower = int(lon // BIN_DEG) * BIN_DEG
                    histogram[frame][lower] += 1
                    for bucket in ((key, frame), ('all', frame), ('all', 'all')):
                        C = counts[bucket]
                        C['positions'] += 1
                        if lat != lon:
                            C['displaced'] += 1
                            C['detected_by_range_check'] += int(abs(lon) > 90)
                            C['missed_by_range_check'] += int(abs(lon) <= 90)
    total = counts[('all', 'all')]
    frozen_totals = frozen['totals']
    assert total['positions'] == frozen_totals['positions'] == 3219
    assert total['detected_by_range_check'] == frozen_totals['naive_axis_out_of_range'] == 1225
    assert total['missed_by_range_check'] == frozen_totals['naive_axis_wrong_but_in_range'] == 1994
    for key, cfg in frozen['configurations'].items():
        got = Counter()
        for frame in ('navarea_iv', 'china_msa'):
            got.update(counts[(key, frame)])
        assert got['positions'] == cfg['positions'], key
        assert got['detected_by_range_check'] == cfg['naive_axis_out_of_range'], key
    out = {}
    for (scope, frame), C in sorted(counts.items()):
        out[f'{scope}|{frame}'] = dict(C) | {
            'missed_share': C['missed_by_range_check'] / C['displaced'] if C['displaced'] else None}
    hist = {frame: {str(lower): n for lower, n in sorted(bins.items())} for frame, bins in histogram.items()}
    assert sum(sum(b.values()) for b in histogram.values()) == 3219
    return {'available': True, 'by_scope_and_frame': out,
            'longitude_histogram': {'bin_degrees': BIN_DEG, 'lower_edge_counts': hist}}


def main():
    inputs = {str(p.relative_to(ROOT)): sha256(p) for p in (OBSERVABLE, TRIAGE, GEO, POSITIONS, MUTATION)}
    result = {
        'description': 'Descriptive decompositions of the inspection and exchange results; no new model calls, labels or scores.',
        'inputs_sha256': inputs,
        'inspection_contract': inspection(),
        'exchange_contract': transposition(),
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    insp = result['inspection_contract']
    print(json.dumps({'pooled_reporting': insp['pooled_reporting'],
                      'criterion_agreement': f"{insp['criterion_agreement_count']}/{insp['criterion_comparisons']}",
                      'transposition_all': result['exchange_contract'].get('by_scope_and_frame', {}).get('all|all'),
                      'transposition_navarea': result['exchange_contract'].get('by_scope_and_frame', {}).get('all|navarea_iv'),
                      'transposition_china': result['exchange_contract'].get('by_scope_and_frame', {}).get('all|china_msa')},
                     indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
