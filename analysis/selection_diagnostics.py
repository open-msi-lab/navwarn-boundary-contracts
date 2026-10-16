"""Text volume selected at the 20% quota (Appendix Table B.1); characters, not review time.

    python analysis/selection_diagnostics.py
"""
from pathlib import Path
import gzip,json,math
from collections import defaultdict
DATA=Path(__file__).resolve().parents[1]/'data'

def expected_sum(scores,values,k):
    groups=defaultdict(list)
    for s,v in zip(scores,values):groups[float(s)].append(v)
    total=0.
    for s in sorted(groups,reverse=True):
        values=groups[s];take=min(k,len(values));total+=sum(values)*take/len(values);k-=take
        if not k:break
    assert k==0
    return total

def main():
    rows=json.loads(gzip.decompress((DATA/'observable-records.json.gz').read_bytes()))['rows']
    scores=json.loads(gzip.decompress((DATA/'triage-scores.json.gz').read_bytes()))
    index={(r['configuration'],r['uid']):r for r in rows}
    fields=['geometry','temporal','lifecycle_refs'];bad=[r for r in rows if r['set']=='Reporting' and
       any(r['counts']['N'][f]['fp']+r['counts']['N'][f]['fn'] for f in fields)]
    out={'scope':'Descriptive review-prompted diagnostic at the fixed 20% record-count quota. Characters are not review time and are not comparable work units across languages.',
         'reporting_core_error_outputs':len(bad),'reporting_core_error_warnings':len({r['uid'] for r in bad}),
         'configurations':{}}
    for c in sorted({r['configuration'] for r in rows}):
        ss=[r for r in scores if r['configuration']==c]
        chars=[round(math.expm1(index[(c,r['uid'])]['features'][0])) for r in ss]
        vals={m:expected_sum([r['scores'][m] for r in ss],chars,round(.2*len(ss)))
              for m in ['incident','text_length','structural_learner','full_learner']}
        out['configurations'][c]={'quota_n':round(.2*len(ss)),'expected_selected_characters':vals,
                                  'full_to_incident_ratio':vals['full_learner']/vals['incident']}
    (DATA/'selection-diagnostics.json').write_text(json.dumps(out,sort_keys=True,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
