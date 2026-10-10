#!/usr/bin/env python3
"""Check existing test-score reversal without retained retries; no model calls."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/runs/heavywater_controls'
PAPER=ROOT/'analysis'
records=json.loads((OUT/'TEST_RECORD_MATCHES.json').read_text())
index={(r['cell'],r['source_id']):r['selected_attempt'] for r in records if r['status']=='MATCH'}
assert len(index)==11800
def auc(m,h):return float(((m[:,None]>h[None,:])+.5*(m[:,None]==h[None,:])).mean())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rows=[];summ=[];hashes={}
for proxy in ['OPT','Falcon']:
    p=PAPER/f'data/sensitivity/cached_{proxy}.npz';hashes[str(p.relative_to(ROOT))]=sha(p)
    with np.load(p) as z:
        q=pd.DataFrame({k:z[k] for k in ['row_id','source_id','dataset','generator','watermark_method','label']});score=z['query_lower']
    humans=q[q.label==0].sort_values(['dataset','source_id','generator','row_id']).drop_duplicates(['dataset','source_id'])
    h={ds:score[g.index] for ds,g in humans.groupby('dataset')}
    for (ds,gen),g in q[(q.label==1)&(q.watermark_method=='HeavyWater')].groupby(['dataset','generator']):
        sel=[i for i,r in g.iterrows() if index[(ds+'::'+gen,r.source_id)]==1]
        assert sel
        rows.append(dict(proxy=proxy,dataset=ds,generator=gen,all_n=len(g),first_attempt_n=len(sel),
                         all_auc=auc(score[g.index],h[ds]),first_attempt_auc=auc(score[sel],h[ds]),
                         all_mean_R=float(score[g.index].mean()),first_attempt_mean_R=float(score[sel].mean())))
    subset=[r for r in rows if r['proxy']==proxy]
    summ.append(dict(proxy=proxy,tasks=len(subset),all_n=sum(r['all_n'] for r in subset),first_attempt_n=sum(r['first_attempt_n'] for r in subset),
        all_macro_auc=float(np.mean([r['all_auc'] for r in subset])),first_attempt_macro_auc=float(np.mean([r['first_attempt_auc'] for r in subset])),
        all_reversed_cells=sum(r['all_auc']<.5 for r in subset),first_attempt_reversed_cells=sum(r['first_attempt_auc']<.5 for r in subset)))
with (OUT/'FIRST_ATTEMPT_BY_CELL.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'FIRST_ATTEMPT_AUDIT.json').write_text(json.dumps(dict(summary=summ,sources=hashes,
    interpretation='Existing score audit: human comparison set unchanged within each domain. First-attempt accepted documents are still selection-conditioned; this is not an unfiltered causal intervention.'),indent=2)+'\n')
print(json.dumps(summ,indent=2))
