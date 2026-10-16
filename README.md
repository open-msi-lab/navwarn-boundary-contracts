# Data for: Assuring large language model extraction of navigational warnings

Data and analysis code for the article *Assuring large language model extraction of
navigational warnings: A boundary-contract framework from free text to geospatial exchange*
(Hongze Liu, Muzhuang Guo, Zhengjiang Liu; Navigation College, Dalian Maritime University;
submitted to *Ocean Engineering*). The repository holds the extraction prompt, the schemas,
the per-record data behind the reported results and the code that recomputes them.

## Quick start

Python 3.12:

```sh
python -m pip install -r requirements.txt
python reproduce.py
```

`reproduce.py` works on a temporary copy, needs no network access or model call, and
exits non-zero if any check fails. It recomputes the results listed below from `data/`,
compares them with the shipped outputs, and compares the numbers printed in the article
with the recomputed values at their printed precision.

## Contents

| Path | Content | Article |
| --- | --- | --- |
| `prompt/extraction-prompt-v1.3.txt` | Extraction prompt sent to every configuration (system message with five synthetic examples; user message template). | §4.3, Appendix C |
| `schemas/wire-schema-v1.2.json` | JSON Schema of the model response. | §4.3 |
| `schemas/record-fields-v1.1.json` | Fields and code lists of the evidence-linked record. | Table 2 |
| `sources/source-index.csv` | The 250 sampled warnings: sample, identifier, language, source, year, warning number, normalization and SHA-256 digest of the normalized text. | §4.1 |
| `sources/navarea-iv-texts.jsonl` | Normalized texts of the 125 sampled NAVAREA IV warnings. | §4.1 |
| `data/per-record-counts.json.gz` | Per-record validity and tp/fp/fn counts per field and matching mode for the 1600 responses and for the 4800 reprocessed records. | Tables 5–7 |
| `data/processing-results.json` | Response-set summaries, processing-policy summaries and adjacent-policy contrasts. | Table 7, Fig. 3 |
| `data/matching-tolerances.json` | Value-equivalence tolerances of the normalized matching mode. | §4.4 |
| `data/aggregates.json` | Corpus and sample sizes, reference agreement, evidence-span F1, pairwise comparisons and rule-parser scores. | §4.1–4.2, Table 5 |
| `data/observable-records.json.gz` | Per-record review features and error indicators. | §5.3 |
| `data/triage-results.json`, `triage-models.json`, `triage-scores.json.gz` | Review scores, fitted models and residual error by policy and quota. | Table 9, Fig. 5 |
| `data/selection-diagnostics.json` | Text volume selected at the 20% quota. | Table B.1 |
| `data/fault-injection.json` | Detection counts of the content checks on injected corruptions. | §5.3, Fig. 4a |
| `data/geospatial-results.json` | Exchange counts per configuration. | Table 10 |
| `data/geometry-positions.json.gz` | Coordinates of every projected geometry (no other record content). | Table 10, Fig. 6 |
| `data/decompositions.json` | Alert quality, error composition, allocation identity and transposition blind band. | Table 8, Table 10, Figs 4 and 6 |
| `data/examples/` | Four English records with source text and their GeoJSON encoding. | Fig. 7, §5.4 |
| `analysis/` | Code that recomputes the results above and checks the printed values. | |

## Source texts

- **NAVAREA IV** warnings were issued by the U.S. National Geospatial-Intelligence Agency.
  The normalized texts of the 125 sampled warnings are in `sources/navarea-iv-texts.jsonl`.
- **China Maritime Safety Administration** warnings are not redistributed. They are
  published at https://www.msa.gov.cn/. The 125 sampled warnings are listed in
  `sources/source-index.csv` by identifier, year, warning number and the SHA-256 digest of
  the normalized text against which the evidence offsets are defined.

The reference annotations and the saved model responses are not released; the per-record
counts in `data/` were computed from them.

## Licence

Code: MIT (`LICENSE`). Data, prompt and schemas: CC BY 4.0. See `LICENSE-DATA.md` for the
source texts.

## Citation

See `CITATION.cff`. Please cite the article once it is published.
