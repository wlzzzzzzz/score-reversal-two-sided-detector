#!/usr/bin/env python3
"""Analyze all frozen controls together, retaining every generated outcome."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/runs/heavywater_controls'
def dump(name,x): (OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def writecsv(name,rows):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def credit(m,h):return (m[:,None]>h[None,:]).astype(float)+.5*(m[:,None]==h[None,:])

def main():
    p=json.loads((OUT/'PROTOCOL.json').read_text());arms=[a['name'] for a in p['arms']];domains=p['domains']
    allrows=[];lookup={};meta=[];hashes={}
    expected=set((r['source_id'],a) for r in json.loads((OUT/'SOURCES.json').read_text()) for a in arms)
    observed=set()
    for j in range(4):
        assert json.loads((OUT/f'shard_{j}/DONE.json').read_text())['status']=='PASS'
        for path in sorted((OUT/f'shard_{j}').glob('*.json')):
            if path.name in ['RUNTIME.json','PROGRESS.json','DONE.json']:continue
            d=json.loads(path.read_text());hashes[str(path.relative_to(OUT))]=sha(path)
            if 'arm' in d:
                name=d['arm']['name'];observed.add((d['source_id'],name))
                meta.append(dict(dataset=d['dataset'],source_id=d['source_id'],arm=name,generated_tokens=len(d['generated_ids']),
                    quality_flagged=bool(d['quality_flags']),quality_flags='|'.join(d['quality_flags']),serialization_changed=d['raw']!=d['serialized']))
                entries=[(name,d['scores'])]
            else:entries=list(d['scores'].items())
            for name,scores in entries:
                lookup[(d['source_id'],name)]=dict(dataset=d['dataset'],scores=scores)
                for view,s in scores.items():
                    if s is None:continue
                    residual=np.array(s['residuals']);var=np.array(s['variances']);n=len(residual)
                    assert n==len(var) and n>=1
                    assert abs(s['full']['R']-residual.sum()/max(np.sqrt(var.sum()),n*1e-9))<1e-10
                    for length in ['full','first64','first128']:
                        z=s[length];allrows.append(dict(dataset=d['dataset'],source_id=d['source_id'],arm=name,view=view,length=length,**z))
    assert observed==expected,(len(observed),len(expected))
    writecsv('DOCUMENT_SCORES.csv',allrows);writecsv('GENERATION_METADATA.csv',meta)
    sources=json.loads((OUT/'SOURCES.json').read_text());ids={ds:[r['source_id'] for r in sources if r['dataset']==ds] for ds in domains}

    def value(sid,name,view,length):
        r=lookup[(sid,name)]['scores'].get(view)
        if not r:return None
        v=r[length]
        if length!='full' and v['n']<int(length[5:]):return None
        return v['R']

    names=arms+['saved_unwatermarked','saved_heavywater'];bydomain=[];summaries=[]
    for view in ['serialized','prompt_conditioned']:
        for length in ['full','first64','first128']:
            for name in names:
                blocks=[]
                for ds in domains:
                    m=np.array([v for sid in ids[ds] if (v:=value(sid,name,view,length)) is not None])
                    h=np.array([v for sid in ids[ds] if (v:=value(sid,'human',view,length)) is not None])
                    if not len(m) or not len(h):continue
                    r=dict(arm=name,view=view,length=length,dataset=ds,machines=len(m),humans=len(h),
                         mean_R=float(m.mean()),human_mean_R=float(h.mean()),auc=float(credit(m,h).mean()))
                    bydomain.append(r);blocks.append(r)
                if blocks:summaries.append(dict(arm=name,view=view,length=length,domains=len(blocks),machines=sum(x['machines'] for x in blocks),
                    humans=sum(x['humans'] for x in blocks),mean_R=float(np.mean([x['mean_R'] for x in blocks])),
                    human_mean_R=float(np.mean([x['human_mean_R'] for x in blocks])),auc=float(np.mean([x['auc'] for x in blocks]))))
    writecsv('BY_DOMAIN.csv',bydomain);writecsv('SUMMARY.csv',summaries)
    rng=np.random.default_rng(20260917);effects=[]
    for view in ['serialized','prompt_conditioned']:
        for length in ['full','first64','first128']:
            for a,b in p['primary_contrasts']:
                dauc=[];dr=[];bdauc=[];bdr=[];sizes=[]
                for ds in domains:
                    common=[sid for sid in ids[ds] if all(value(sid,z,view,length) is not None for z in [a,b,'human'])]
                    n=len(common)
                    if n<2:continue
                    aa=np.array([value(sid,a,view,length) for sid in common]);bb=np.array([value(sid,b,view,length) for sid in common]);hh=np.array([value(sid,'human',view,length) for sid in common])
                    diff=credit(aa,hh)-credit(bb,hh);delta=aa-bb
                    counts=rng.multinomial(n,[1/n]*n,size=5000)/n
                    bdauc.append(np.einsum('bi,ij,bj->b',counts,diff,counts));bdr.append(counts@delta)
                    dauc.append(diff.mean());dr.append(delta.mean());sizes.append(n)
                if not sizes:continue
                ba=np.mean(bdauc,axis=0);br=np.mean(bdr,axis=0)
                effects.append(dict(arm=a,reference=b,view=view,length=length,domains=len(sizes),paired_sources=sum(sizes),
                    delta_auc=float(np.mean(dauc)),auc_ci_low=float(np.quantile(ba,.025)),auc_ci_high=float(np.quantile(ba,.975)),
                    delta_mean_R=float(np.mean(dr)),R_ci_low=float(np.quantile(br,.025)),R_ci_high=float(np.quantile(br,.975))))
    writecsv('PAIRED_EFFECTS.csv',effects)
    # Same generated IDs: isolate detector context and text serialization effects.
    shifts=[]
    for name in arms:
        for va,vb in [('serialized','raw_ids'),('prompt_conditioned','raw_ids')]:
            values=[]
            for ds in domains:
                local=[]
                for sid in ids[ds]:
                    a=value(sid,name,va,'full');b=value(sid,name,vb,'full')
                    if a is not None and b is not None:local.append(a-b)
                if local:values.append(float(np.mean(local)))
            shifts.append(dict(arm=name,view=va,reference_view=vb,domains=len(values),delta_mean_R=float(np.mean(values))))
    writecsv('SCORING_VIEW_EFFECTS.csv',shifts)
    dump('ANALYSIS_AUDIT.json',dict(status='PASS',generated_documents=len(observed),expected_documents=len(expected),sources=len(sources),
        source_domains=domains,new_test_generations=False,bootstrap_draws=5000,bootstrap_seed=20260917,
        bootstrap_unit='paired prompt/source, resampled within each of four calibration domains; human and all compared machine arms share each source draw',
        intervals_scope='Exploratory percentile intervals conditional on the selected domains, model and one generation seed per source; not 60-cell benchmark intervals',
        missing_score_entries=len(expected)*3-sum(len([v for v in lookup[(sid,a)]['scores'].values() if v is not None]) for sid,a in expected),
        quality_flagged_but_retained=sum(r['quality_flagged'] for r in meta),inputs=hashes,analysis_script_sha256=sha(__file__)))
    print('PRIMARY SUMMARY')
    for r in summaries:
        if r['view']=='serialized' and r['length']=='full':print(r)
    print('PRIMARY CONTRASTS')
    for r in effects:
        if r['view']=='serialized' and r['length']=='full':print(r)

if __name__=='__main__':main()
