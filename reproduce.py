"""Recompute the reported results from data/ and compare them with the shipped outputs and the printed values.

    python reproduce.py

Runs on a temporary copy, so the shipped files are never modified. Needs no
network access, source text or model call. Writes reproduction-report.json and
exits non-zero if any check fails.
"""
from pathlib import Path
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
FIT_TOL, SCORE_TOL = 1e-7, 1e-8


def read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)


def max_numeric_difference(a, b):
    if isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b)
        return max((max_numeric_difference(a[k], b[k]) for k in a), default=0.0)
    if isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b)
        return max((max_numeric_difference(x, y) for x, y in zip(a, b)), default=0.0)
    if type(a) in (float, int) and type(b) in (float, int):
        return abs(a - b)
    assert a == b, (a, b)
    return 0.0


def run(script, cwd, env):
    p = subprocess.run([sys.executable, '-I', f'analysis/{script}'], cwd=cwd, env=env, text=True,
                       capture_output=True)
    assert p.returncode == 0, (script, p.stdout[-2000:], p.stderr[-4000:])
    lines = p.stdout.strip().splitlines()
    return json.loads(lines[-1]) if lines and lines[-1].startswith('{') else None


def main():
    report = {'passed': False}
    with tempfile.TemporaryDirectory(prefix='repro-') as td:
        tmp = Path(td)
        for name in ('analysis', 'data'):
            shutil.copytree(HERE / name, tmp / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')

        # Content and processing: summaries, contrasts, intervals and Holm adjustment.
        report['content_and_processing'] = run('content_processing.py', tmp, env)['checked']

        # Inspection: review scores and allocations, then text volume (Table B.1).
        run('triage.py', tmp, env)
        run('selection_diagnostics.py', tmp, env)
        data, shipped = tmp / 'data', HERE / 'data'
        exact = {p: (data / p).read_bytes() == (shipped / p).read_bytes()
                 for p in ('triage-results.json', 'selection-diagnostics.json')}
        assert all(exact.values()), exact
        fits = max_numeric_difference(read(data / 'triage-models.json'), read(shipped / 'triage-models.json'))
        scores = max_numeric_difference(read(data / 'triage-scores.json.gz'), read(shipped / 'triage-scores.json.gz'))
        assert fits <= FIT_TOL and scores <= SCORE_TOL, (fits, scores)
        report['inspection'] = {'exact_outputs': exact, 'max_fit_difference': fits, 'max_score_difference': scores}

        # Alert quality, error composition and the transposition blind band.
        run('decompositions.py', tmp, env)
        fresh, frozen = read(data / 'decompositions.json'), read(shipped / 'decompositions.json')
        for part in ('inspection_contract', 'exchange_contract'):
            assert fresh[part] == frozen[part], part
        report['decompositions_identical'] = True

        # Exchange: the four English example chains.
        report['example_chains'] = run('examples.py', tmp, env)['geometries_decoded']

        # Printed numbers, from the recomputed files.
        report['printed_values_matched'] = run('printed_values.py', tmp, env)['printed_values_matched']
    report['passed'] = True
    report['scope'] = ('Recomputes the content summaries, processing contrasts, inspection allocations, '
                       'alert-quality, error-composition and exchange results from the per-record data and '
                       'compares them with the shipped outputs and the printed values. No model call is made '
                       'and no source text is retrieved; the per-record counts were produced from reference '
                       'annotations and saved responses that are not released.')
    (HERE / 'reproduction-report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
