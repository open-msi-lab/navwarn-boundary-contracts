"""Check the four English example chains: source text -> record -> GeoJSON -> decoded geometry.

For each example, the source digest must match the record, the encoder must
reproduce the shipped GeoJSON exactly, and the consumer-side decoding must
return every geometry of the record in order.

    python analysis/examples.py
"""
from pathlib import Path
import hashlib
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exchange import consume, convert  # noqa: E402

EXAMPLES = Path(__file__).resolve().parents[1] / 'data/examples'


def main():
    records = json.loads((EXAMPLES / 'english-replay-records.json').read_text())
    geojson = json.loads((EXAMPLES / 'english-geojson-examples.json').read_text())
    assert set(records) == set(geojson) and len(geojson) == 4
    out = {}
    for key, entry in sorted(records.items()):
        record, text = entry['record'], entry['source']
        assert hashlib.sha256(text.encode()).hexdigest() == record['text_sha256'], key
        assert convert(record, text, lambda *_: []) == geojson[key], key
        out[key] = consume(geojson[key], record)
    print(json.dumps({'passed': True, 'geometries_decoded': out}))


if __name__ == '__main__':
    main()
