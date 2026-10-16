"""Recompute the content and processing aggregates from per-record counts.

Reads the per-record tp/fp/fn counts of the 1600 responses and of the 4800
reprocessed records (three processing policies), recomputes every response-set
summary and every adjacent-policy contrast, and compares them with
data/processing-results.json.

    python analysis/content_processing.py

Prints a JSON report; exits non-zero on the first mismatch.
"""
from pathlib import Path
import gzip
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scoring import summary, paired, permutation_p, pool, holm_adjust, MODES  # noqa: E402

PRIMARY_ALPHA = 0.05 / 8   # eight primary contrasts, Bonferroni


def is_primary(key):
    """Full-pipeline configurations on the reporting sample form the primary family."""
    return key.endswith('/full')


def main():
    per = json.loads(gzip.decompress((ROOT / 'data/per-record-counts.json.gz').read_bytes()))
    frozen = json.loads((ROOT / 'data/processing-results.json').read_text())
    checked = {'response_set_summaries': 0, 'policy_summaries': 0, 'contrasts': 0}
    primary = []
    for key, evals in per['original'].items():
        assert summary(evals) == frozen['response_sets'][key], ('response-set summary', key)
        checked['response_set_summaries'] += 1
    for key, policies in per['derived'].items():
        for name, evals in policies.items():
            assert summary(evals) == frozen['processing'][key]['policies'][name], ('policy summary', key, name)
            checked['policy_summaries'] += 1
        for contrast, ref in frozen['processing'][key]['contrasts'].items():
            after, before = contrast.split('-minus-')
            ids = sorted(policies[before])
            for m in MODES:
                av = [pool(policies[after][u]['counts'][m])['f1'] for u in ids]
                bv = [pool(policies[before][u]['counts'][m])['f1'] for u in ids]
                r = ref[m]
                assert abs(sum(av) / len(av) - sum(bv) / len(bv) - r['delta_item_f1']) < 1e-12, (key, contrast, m)
                assert paired(av, bv) == r['ci95'], ('ci95', key, contrast, m)
                alpha = PRIMARY_ALPHA if is_primary(key) else 0.05
                assert paired(av, bv, alpha=alpha) == r['adjusted_ci'], ('adjusted ci', key, contrast, m)
                assert permutation_p(av, bv) == r['permutation'], ('sign-flip', key, contrast, m)
                checked['contrasts'] += 1
            if is_primary(key):
                primary.append(ref['S'])
    assert len(primary) == 8
    for r, p in zip(primary, holm_adjust([r['permutation']['p'] for r in primary])):
        assert r['holm_p'] == p, ('holm', r['holm_p'], p)
    print(json.dumps({'passed': True, 'checked': checked}))


if __name__ == '__main__':
    main()
