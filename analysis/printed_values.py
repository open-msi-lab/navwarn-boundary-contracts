"""Compare the numbers printed in the manuscript with values recomputed from data/.

Every value below is copied from the manuscript and compared after rounding to
the printed precision: Sections 4.1-4.2, Tables 5-10, Sections 5.1-5.4 and
Appendix Tables B.1-B.2.

    python analysis/printed_values.py
"""
from pathlib import Path
import gzip
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scoring import paired, permutation_p, pool, holm_adjust, FIELDS  # noqa: E402

CONFIGS = {'Sol': 'Sol/full', 'Luna': 'Luna/full', 'Flash': 'Flash/full', 'DeepSeek': 'DeepSeek/full'}
SIX = tuple(f for f in FIELDS if f not in ('restrictions', 'affected_entities'))

# Table 5: valid, S micro, S message, N micro, English-75 micro over 8 and 6 fields.
TABLE5 = {'Sol': (150, '0.884', '0.892', '0.896', '0.934', '0.941'),
          'Luna': (150, '0.782', '0.841', '0.802', '0.744', '0.888'),
          'Flash': (150, '0.842', '0.871', '0.875', '0.852', '0.833'),
          'DeepSeek': (148, '0.800', '0.844', '0.821', '0.808', '0.909')}
# Table 6: strict value F1 by field (in FIELDS order) and by source frame (en, zh).
TABLE6 = {'Sol': ('0.980', '0.973', '0.928', '0.985', '0.929', '0.788', '0.656', '0.857', '0.934', '0.845'),
          'Luna': ('0.967', '0.953', '0.896', '0.935', '0.888', '0.558', '0.774', '0.353', '0.744', '0.813'),
          'Flash': ('0.987', '0.933', '0.720', '0.975', '0.898', '0.771', '0.770', '0.758', '0.852', '0.833'),
          'DeepSeek': ('0.993', '0.920', '0.835', '0.884', '0.868', '0.561', '0.810', '0.833', '0.808', '0.793')}
# Table 7: valid / strict message F1 per policy; repair - abstain with interval; transitions.
TABLE7 = {'Sol/full': ('137 / 0.8804', '147 / 0.8807', '150 / 0.8918', '0.0111', '0.0000', '0.0378', (5, 3, 0)),
          'Luna/full': ('142 / 0.8391', '148 / 0.8397', '150 / 0.8409', '0.0012', '-0.0021', '0.0085', (1, 6, 0)),
          'Flash/full': ('145 / 0.8666', '146 / 0.8666', '150 / 0.8706', '0.0041', '0.0000', '0.0123', (3, 6, 0)),
          'DeepSeek/full': ('135 / 0.8215', '139 / 0.8229', '148 / 0.8437', '0.0208', '0.0056', '0.0426', (22, 13, 0)),
          'Sol/validation': ('85 / 0.8689', '100 / 0.8690', '100 / 0.8645', '-0.0045', '-0.0088', '-0.0008', (0, 0, 5))}


def f3(x):
    return f'{x:.3f}'


def f4(x):
    return f'{x:.4f}'.replace('-0.0000', '0.0000')


def micro(evals, mode, fields=FIELDS, frame=None):
    sums = {f: {k: sum(e['counts'][mode][f][k] for e in evals.values()
                       if frame is None or e['source_frame'] == frame) for k in ('tp', 'fp', 'fn')} for f in fields}
    return pool(sums, fields)['f1']


def message_f1(evals, mode='S'):
    return sum(pool(e['counts'][mode])['f1'] for e in evals.values()) / len(evals)


def main():
    per = json.loads(gzip.decompress((DATA / 'per-record-counts.json.gz').read_bytes()))
    stored = json.loads((DATA / 'processing-results.json').read_text())
    original, derived = per['original'], per['derived']
    n = 0

    def same(got, printed, what):
        nonlocal n
        assert got == printed, (what, got, printed)
        flat = [x for v in printed for x in (v if isinstance(v, tuple) else (v,))] if isinstance(printed, tuple) else [printed]
        n += sum(x is not None for x in flat)

    for name, key in CONFIGS.items():
        ev = original[key]
        valid, s_micro, s_msg, n_micro, en8, en6 = TABLE5[name]
        same(sum(e['valid'] for e in ev.values()), valid, (name, 'valid'))
        same(f3(micro(ev, 'S')), s_micro, (name, 'S micro'))
        same(f3(message_f1(ev)), s_msg, (name, 'S message'))
        same(f3(micro(ev, 'N')), n_micro, (name, 'N micro'))
        same(f3(micro(ev, 'S', frame='en')), en8, (name, 'English 8 fields'))
        same(f3(micro(ev, 'S', SIX, frame='en')), en6, (name, 'English 6 fields'))
        for field, printed in zip(FIELDS, TABLE6[name][:8]):
            same(f3(micro(ev, 'S', (field,))), printed, (name, field))
        same(f3(micro(ev, 'S', frame='en')), TABLE6[name][8], (name, 'NAVAREA IV frame'))
        same(f3(micro(ev, 'S', frame='zh')), TABLE6[name][9], (name, 'China MSA frame'))

    # Table 5 columns held as aggregates: evidence-span F1, pairwise comparisons and the
    # rule parser on the 75 English warnings (six-field score pooled from its per-field counts).
    ext = json.loads((DATA / 'aggregates.json').read_text())
    arm = CONFIGS
    same(tuple(f3(ext['evidence_span_exact_f1'][arm[c]]['f1']) for c in CONFIGS),
         ('0.613', '0.572', '0.553', '0.489'), 'Table 5 span F1')
    pairs = {(p['a'], p['b']): p for p in ext['pairwise_message_f1']['pairs']}
    for name, key in CONFIGS.items():   # the stored per-item means equal the recomputed message F1
        means = [p['mean_a'] for (a, _), p in pairs.items() if a == arm[name]] + \
                [p['mean_b'] for (_, b), p in pairs.items() if b == arm[name]]
        assert means and all(abs(m - message_f1(original[key])) < 1e-12 for m in means), name
    for b, printed in (('Luna/full', ('0.051', '0.015', '0.090')),
                       ('DeepSeek/full', ('0.048', '0.001', '0.099'))):
        d = pairs[('Sol/full', b)]['paired_diff_bonferroni']
        same((f3(d['point']), f3(d['lo']), f3(d['hi'])), printed, ('pairwise Sol vs', b))
    rp = ext['rule_parser_english_75']
    pf = rp['per_field_value_counts']
    six = {k: sum(pf[f][k] for f in SIX) for k in ('tp', 'fp', 'fn')}
    same((f3(rp['micro_value_f1']), f3(2 * six['tp'] / (2 * six['tp'] + six['fp'] + six['fn'])), rp['n_labels']),
         ('0.507', '0.671', 75), 'rule parser on English 75')

    # Sections 4.1 and 4.2 (aggregates).
    c, fr, ra = ext['corpora'], ext['sampling_frames'], ext['reference_agreement']
    same((c['navarea_iv']['messages'], tuple(c['navarea_iv']['years']), c['china_msa']['documents'],
          tuple(c['china_msa']['warning_or_notice_years']), c['china_msa']['warning_or_notice'],
          c['china_msa']['by_type']['B'], c['china_msa']['pool']),
         (14274, (2012, 2024), 151512, (2015, 2026), 125656, 25856, 125251), 'corpus sizes')
    same((fr['navarea_iv']['seed'], fr['navarea_iv']['n_final'], fr['navarea_iv']['n_final'] - fr['navarea_iv']['n_base_drawn'],
          fr['china_msa']['seed'], fr['china_msa']['n_final'], fr['china_msa']['n_final'] - fr['china_msa']['n_base_drawn']),
         (20260704, 573, 23, 20260704, 385, 25), 'sampling frames')
    b = ext['additional_sample_body_language']
    same((b['en'], b['zh'], b['bilingual']), (58, 41, 1), 'additional-sample body languages')
    sa = ext['reporting_split_audit']
    same((sa['exact_identifier_overlap'], sa['reporting_items_sharing_dev_template_cluster']), (0, 6), 'split audit')
    same((f3(ra['kappa']['message_function']), f3(ra['kappa']['event_type']), f3(ra['kappa']['authority']),
          f3(ra['evidence_span_f1']['exact']), f3(ra['evidence_span_f1']['relaxed']), f3(ra['lifecycle_refs_f1']),
          f3(ra['gamma']['mean']), ra['gamma']['n'], ra['gamma']['undefined'],
          ra['dual_agreed'], ra['adjudicated'], ra['decision_rows']),
         ('0.559', '0.285', '0.614', '0.445', '0.648', '0.308', '0.565', 146, 4, 51, 99, 155),
         'reference agreement and adjudication')

    # Section 4.2: reference instances (tp + fn under S) and the warnings that contain them.
    ref = {}
    for f in ('geometry', 'temporal', 'restrictions', 'affected_entities', 'authority', 'lifecycle_refs'):
        units = [e['counts']['S'][f]['tp'] + e['counts']['S'][f]['fn'] for e in original['Sol/full'].values()]
        ref[f] = (sum(units), sum(u > 0 for u in units))
    same((ref['geometry'], ref['temporal'][1], ref['restrictions'], ref['affected_entities'], ref['authority'][1],
          ref['lifecycle_refs']), ((154, 96), 101, (153, 87), (377, 131), 62, (48, 33)), 'reference composition')
    c = original['Sol/full']['2023-726']['counts']['S']['affected_entities']
    same(c['tp'] + c['fn'], 78, 'offshore structures in warning 2023-726')

    # Section 5.1 text.
    sol_add = original['Sol/validation']
    same((f3(micro(sol_add, 'S')), f3(message_f1(sol_add)), sum(e['valid'] for e in sol_add.values())),
         ('0.847', '0.864', 100), 'Sol additional sample')
    same(f3(micro(original['Flash/full'], 'N', ('geometry',))), '0.917', 'Flash geometry under N')
    units = {u: e['counts']['N']['geometry']['tp'] + e['counts']['N']['geometry']['fn']
             for u, e in original['Sol/full'].items()}
    same((sum(units.values()), units['2023-726'], f'{100 * units["2023-726"] / sum(units.values()):.1f}'),
         (238, 78, '32.8'), 'normalized reference geometry units')

    # Table 7 and Section 5.2 text.
    for key, (res, abst, rep, delta, lo, hi, transitions) in TABLE7.items():
        pol = derived[key]
        for printed, name in ((res, 'resolve'), (abst, 'abstain'), (rep, 'repair')):
            got = f"{sum(e['valid'] for e in pol[name].values())} / {f4(message_f1(pol[name]))}"
            same(got, printed, (key, name))
        ids = sorted(pol['abstain'])
        av = [pool(pol['repair'][u]['counts']['S'])['f1'] for u in ids]
        bv = [pool(pol['abstain'][u]['counts']['S'])['f1'] for u in ids]
        ci = paired(av, bv, alpha=0.05 / 8 if key.endswith('/full') else 0.05)
        same((f4(ci['point']), f4(ci['lo']), f4(ci['hi'])), (delta, lo, hi), (key, 'repair - abstain'))
        t = stored['processing'][key]['contrasts']['repair-minus-abstain']['S']['field_transitions']
        same((t.get('corrected', 0), t.get('changed_still_wrong', 0), t.get('damaged', 0)), transitions,
             (key, 'transitions'))
    abstain_intervals = []
    primary_p = []
    for key in ('DeepSeek/full', 'Flash/full', 'Luna/full', 'Sol/full'):
        pol = derived[key]
        ids = sorted(pol['resolve'])
        for after, before in (('abstain', 'resolve'), ('repair', 'abstain')):
            av = [pool(pol[after][u]['counts']['S'])['f1'] for u in ids]
            bv = [pool(pol[before][u]['counts']['S'])['f1'] for u in ids]
            if after == 'abstain':
                abstain_intervals.append(paired(av, bv, alpha=0.05 / 8))
            primary_p.append((key, after, permutation_p(av, bv)['p']))
    same(all(-0.005 <= c['lo'] and c['hi'] <= 0.005 for c in abstain_intervals), True,
         'abstain - resolve intervals within +-0.005')
    holm = dict(zip([(k, a) for k, a, _ in primary_p], holm_adjust([p for _, _, p in primary_p])))
    same(f4(holm[('DeepSeek/full', 'repair')]), '0.0020', 'DeepSeek Holm-adjusted sign-flip p')
    changes = stored['processing']['DeepSeek/full']['contrasts']['repair-minus-abstain']['S']['changes']
    same(sum(c['transition'] == 'corrected' and c['parse_failure'] for c in changes), 20,
         'DeepSeek corrections through whole-record fallback')
    deltas = stored['processing']['Sol/validation']['contrasts']['repair-minus-abstain']['S']['count_deltas']
    same((deltas.get('fp', 0), deltas.get('tp', 0)), (5, 0), 'Sol additional: repair adds 5 FP, 0 TP')
    # Tables 8-10.
    insp = json.loads((DATA / 'decompositions.json').read_text())['inspection_contract']
    triage = json.loads((DATA / 'triage-results.json').read_text())
    geo = json.loads((DATA / 'geospatial-results.json').read_text())
    exch = json.loads((DATA / 'decompositions.json').read_text())['exchange_contract']

    def lr(x):
        return '∞' if x == 'inf' else f'{x:.2f}'

    def alerts(block):
        a, c = block['incident'], block['any_content_flag']
        return (f3(a['natural_recall']), f3(a['false_alarm_rate']), lr(a['positive_likelihood_ratio']),
                f3(c['natural_recall']), f3(c['false_alarm_rate']), lr(c['positive_likelihood_ratio']))

    TABLE8 = {'Sol/full': ('9 / 150', '0.222', '0.078', '2.85', '0.000', '0.213', '0.00'),
              'Luna/full': ('32 / 150', '0.281', '0.051', '5.53', '0.250', '0.203', '1.23'),
              'Flash/full': ('12 / 150', '0.500', '0.000', '∞', '0.167', '0.210', '0.79'),
              'DeepSeek/full': ('22 / 150', '0.636', '0.023', '27.15', '0.091', '0.211', '0.43'),
              'Sol/validation': ('12 / 100', '0.167', '0.148', '1.13', '0.250', '0.295', '0.85')}
    for key, row in TABLE8.items():
        s = insp['per_set'][key]
        same((f"{s['core_errors']} / {s['n']}",) + alerts(s['alerts']), row, (key, 'Table 8'))
    pooled = insp['pooled_reporting']
    same((f"{pooled['core_errors']} / {pooled['n']}",) + alerts(pooled),
         ('75 / 600', '0.413', '0.038', '10.85', '0.160', '0.210', '0.76'), 'Table 8 pooled')
    cov = insp['injected_corruption_coverage']['grammar_corrected_checker']
    same((cov['geometry']['newly_triggered_alerts'], cov['geometry']['cases'],
          f"{100 * cov['geometry']['new_alert_coverage']:.1f}", cov['temporal']['newly_triggered_alerts'],
          cov['temporal']['cases']), (55, 67, '82.1', 30, 64), 'injected corruption coverage')
    same((insp['criterion_agreement_count'], insp['criterion_comparisons']), (10, 10), 'LR criterion signs')

    TABLE9 = {'uniform': ('6.00', '21.33', '8.00', '14.67', '12.00'),
              'incident': ('5.11', '17.04', '4.17', '6.02', '11.76'),
              'content_first': ('7.50', '20.42', '8.39', '15.98', '12.46'),
              'incident_first': ('5.83', '16.04', '3.71', '5.92', '11.96'),
              'text_length': ('3.33', '15.83', '3.33', '11.67', '11.25'),
              'structural_learner': ('4.17', '12.50', '1.67', '5.83', '10.00'),
              'full_learner': ('4.17', '11.67', '1.67', '5.83', '10.00'),
              'oracle': ('0.00', '1.67', '0.00', '0.00', '0.00')}
    sets = ('Sol/full', 'Luna/full', 'Flash/full', 'DeepSeek/full', 'Sol/validation')
    for method, row in TABLE9.items():
        got = tuple(f"{100 * triage['configurations'][c]['core'][method]['0.2']['residual']:.2f}" for c in sets)
        same(got, row, (method, 'Table 9'))
    pp = lambda x: f'{100 * x:.2f}'.replace('-0.00', '0.00')  # noqa: E731
    got = tuple((pp(triage['primary_contrasts'][c]['point']),) + tuple(pp(v) for v in triage['primary_contrasts'][c]['family_ci99'])
                for c in sets)
    same(got, (('-0.94', '-3.76', '2.19'), ('-5.37', '-9.67', '0.00'), ('-2.50', '-6.02', '0.42'),
               ('-0.18', '-2.70', '1.39'), ('-1.76', '-6.19', '3.75')), 'Table 9 full - incident')

    T = geo['totals']
    attempted = sum(c['attempted_records'] for c in geo['configurations'].values())
    g = geo['geometry_types']
    same((attempted, T['accepted_records'], T['geometry_features'], T['null_geometry_records'],
          g['POINT'], g['MULTIPOINT'], g['LINE'], g['POLYGON'], T['multi_geometry_records'], T['positions'],
          T['source_spans_in_geometry'], T['rings_closed'], T['rings_reoriented'], T['width_attribute_features']),
         (700, 698, 758, 257, 303, 12, 163, 280, 86, 3219, 1426, 260, 218, 27), 'Table 10 counts')
    band = exch['by_scope_and_frame']
    same(tuple(f"{band[k]['missed_by_range_check']} / {band[k]['displaced']} "
               f"({100 * band[k]['missed_share']:.1f}%)" for k in ('all|all', 'all|navarea_iv', 'all|china_msa')),
         ('1994 / 3219 (61.9%)', '1994 / 2268 (87.9%)', '0 / 951 (0.0%)'), 'Table 10 blind band')
    # Appendix Tables B.1 and B.2.
    sel = json.loads((DATA / 'selection-diagnostics.json').read_text())
    TABLEB1 = {'Sol/full': (30, '16,747', '29,464', '1.76'), 'Luna/full': (30, '21,472', '31,800', '1.48'),
               'Flash/full': (30, '21,264', '31,953', '1.50'), 'DeepSeek/full': (30, '26,089', '32,281', '1.24'),
               'Sol/validation': (20, '6,449', '9,326', '1.45')}
    for key, row in TABLEB1.items():
        c = sel['configurations'][key]
        x = c['expected_selected_characters']
        same((c['quota_n'], f"{x['incident']:,.0f}", f"{x['full_learner']:,.0f}", f"{c['full_to_incident_ratio']:.2f}"),
             row, (key, 'Table B.1'))
    fold = triage['fold_sensitivity']
    rng = lambda d: f'{100 * min(d.values()):.2f}--{100 * max(d.values()):.2f}'  # noqa: E731
    TABLEB2 = {'Sol/full': ('3.33--5.00', '4.17--5.00', '46.72', '50.83', '44.17'),
               'Luna/full': ('10.83--12.50', '10.00--12.50', '59.26', '61.09', '56.67'),
               'Flash/full': ('0.83--1.67', '0.83--1.67', '51.39', '53.76', '51.67'),
               'DeepSeek/full': ('5.83--5.83', '5.83--5.83', '51.88', '57.55', '53.33'),
               'Sol/validation': (None, None, '58.82', '60.18', '57.50')}
    for key, row in TABLEB2.items():
        five = triage['configurations'][key]['five']
        got = ((rng(fold['structural_learner'][key]), rng(fold['full_learner'][key])) if key in fold['full_learner']
               else (None, None)) + tuple(f"{100 * five[m]['0.2']['residual']:.2f}"
                                          for m in ('incident', 'content_first', 'full_learner'))
        same(got, row, (key, 'Table B.2'))
    print(json.dumps({'passed': True, 'printed_values_matched': n}))


if __name__ == '__main__':
    main()
