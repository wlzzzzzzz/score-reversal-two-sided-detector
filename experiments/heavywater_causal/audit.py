#!/usr/bin/env python3
"""Read-only provenance audit and saved-prefix state decomposition."""
from pathlib import Path
import collections
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/runs/heavywater_controls'
BASE=ROOT/'scratch/clean80_english20_test60_plus3wm'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ht(s): return hashlib.sha256(s.encode()).hexdigest()
def dump(name,x): (OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def provenance():
    manifest=BASE/'benchmark/source_manifest.json'
    d=json.loads(manifest.read_text())
    prep=json.loads((BASE/'dataset/preparation_manifest.json').read_text())
    reused={c['cell']:c.get('reuse_source') for c in prep['cells']}
    records=[];aggregates=[];configs=[]
    # Match every current test-cell HeavyWater text to its retained generation record.
    for c in d['cells']:
        if c['split']!='test':continue
        ds,g=c['dataset'],c['generator'];data=json.loads(Path(c['data_path']).read_text())
        pool={}
        recordroot=ROOT/'scratch/broad_watermark_extension' if reused[c['cell']] else BASE/'extension'
        for p in (recordroot/f'records/{ds}/{g}/HeavyWater').glob('*.json'):
            r=json.loads(p.read_text())
            assert ht(r['continuation'])==r['continuation_sha256']
            pool.setdefault(r['continuation_sha256'],[]).append((p,r))
        matched=0;attempts=collections.Counter();reasons=collections.Counter();tokens=[]
        for i,tx in enumerate(data['HeavyWater']):
            found=pool.get(ht(tx),[])
            if not found:
                records.append(dict(cell=c['cell'],row=i,status='NO_EXACT_RECORD_MATCH'));continue
            p,r=found[0];matched+=1;attempts[r['selected_attempt']]+=1
            if r['source_id']=='wp:11':
                matched-=1;attempts[r['selected_attempt']]-=1
                continue
            selected=[a for a in r['attempts'] if a['attempt']==r['selected_attempt']][0]
            tokens.append(selected['diagnostics']['generated_tokens'])
            for a in r['attempts']:
                for reason in a['reasons']:reasons[reason]+=1
            records.append(dict(cell=c['cell'],row=i,status='MATCH',record=str(p.relative_to(ROOT)),record_sha256=sha(p),
                source_id=r['source_id'],slot_index=r['slot_index'],prompt_sha256=r['prompt_sha256'],selected_attempt=r['selected_attempt'],generated_tokens=tokens[-1]))
        aggregates.append(dict(cell=c['cell'],expected=len(data['HeavyWater'])-(ds=='wp'),matched=matched,recordroot=str(recordroot.relative_to(ROOT)),selected_attempt_counts=dict(attempts),rejected_reasons=dict(reasons),
            generated_tokens_mean=sum(tokens)/len(tokens) if tokens else None,at_max200=sum(x==200 for x in tokens)))
    paths=list((BASE/'extension/generation_audit/HeavyWater').glob('*/*/run_config.json'))+list((ROOT/'scratch/broad_watermark_extension/generation_audit/HeavyWater').glob('*/*/run_config.json'))
    for p in sorted(paths):
        r=json.loads(p.read_text());configs.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),**{k:r.get(k) for k in ['generator','datasets','repetition_penalty','no_repeat_ngram_size','strict_text_quality','max_new_tokens','token_min_new_tokens','base_seed','shard_count','shard_index','torch','python']}))
    old=json.loads((ROOT/'results/runs/heavywater_prefix/FROZEN.json').read_text())
    codeparity={p:sha(ROOT/p)==h for p,h in old['code_sha256'].items()}
    result=dict(test_machine_texts=sum(a['expected'] for a in aggregates),exact_record_matches=sum(a['matched'] for a in aggregates),
        all_selected_first_attempt=sum(r.get('selected_attempt')==1 for r in records if r['status']=='MATCH'),
        code_matches_previous_frozen_replay=codeparity,generation_configs=configs,cells=aggregates,
        limitations=['Run configs do not persist fully resolved historical Hugging Face GenerationConfig.',
         'Explicit top_k=0 is established by frozen HeavyWater worker code. Baseline top_k=50 remains a runtime reconstruction because the baseline call omits top_k.',
         'Original generated token IDs and rejected candidate texts are not retained in historical HeavyWater records.',
         'No new scoring or tuning on the test set: only record/hash/metadata inspection.'])
    dump('PROVENANCE_AUDIT.json',result)
    dump('TEST_RECORD_MATCHES.json',records)
    print({k:result[k] for k in ['test_machine_texts','exact_record_matches','all_selected_first_attempt','code_matches_previous_frozen_replay']})

def states():
    import numpy as np
    import torch
    old=ROOT/'results/runs/heavywater_prefix'
    values=torch.load(old/'all_state_shifts.pt',map_location='cpu',weights_only=False)
    snaps=torch.load(old/'logits_snapshots.pt',map_location='cpu',weights_only=False)
    assert len(values)==len(snaps)==25
    result={}
    for group in ['HeavyWater','unwatermarked']:
        pairs=[(x,y) for x,y in zip(values,snaps) if x['metadata']['mechanism']==group]
        assert all(x['metadata']==y['metadata'] for x,y in pairs)
        raw=np.stack([x['all_state_delta_from_nucleus'] for x,y in pairs]).astype(float)
        # Exact uniform-state centering removes numerical OT mean.
        D=raw-raw.mean(axis=1,keepdims=True)
        ids=np.array([y['selected_state_id'] for x,y in pairs])
        prev=np.array([int(y['full_prefix_ids'][-1]) for x,y in pairs])
        n,k=D.shape;mu=D.mean(axis=0);freq=np.bincount(ids,minlength=k)/n
        obs=D[np.arange(n),ids].mean()
        occupancy=float(((freq-1/k)*mu).sum())
        indicators=np.eye(k)[ids]
        covariance=float(((indicators-freq)*(D-mu)).mean(axis=0).sum())
        assert abs(obs-occupancy-covariance)<1e-12
        rng=np.random.default_rng(20260917)
        independent=rng.integers(0,k,size=(20000,n))
        independent_means=D[np.arange(n)[None,:],independent].mean(axis=1)
        unique,inv=np.unique(prev,return_inverse=True)
        # Re-randomize one state per distinct previous-token ID to preserve state reuse.
        assigned=rng.integers(0,k,size=(20000,len(unique)))[:,inv]
        keyed_means=D[np.arange(n)[None,:],assigned].mean(axis=1)
        def summary(a):return dict(mean=float(a.mean()),q025=float(np.quantile(a,.025)),q975=float(np.quantile(a,.975)),
                                   observed_percentile=float((a<=obs).mean()))
        result[group]=dict(prefixes=n,unique_sources=len(set(x['metadata']['source_id'] for x,y in pairs)),unique_previous_tokens=len(unique),
            selected_mean_nats=float(obs),negative_selected_count=int((D[np.arange(n),ids]<0).sum()),
            average_uniform_state_negative_fraction=float((D<0).mean()),occupancy_term=occupancy,state_context_covariance_term=covariance,
            independent_state_counterfactual=summary(independent_means),reused_state_counterfactual=summary(keyed_means),
            interpretation='Empirical decomposition on fixed saved prefixes. Monte Carlo percentiles are descriptive, not a population causal test on independent prefixes.')
    dump('SAVED_PREFIX_STATE_AUDIT.json',dict(groups=result,sources={str((old/x).relative_to(ROOT)):sha(old/x) for x in ['all_state_shifts.pt','logits_snapshots.pt']}))
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    if sys.argv[1]=='provenance':provenance()
    else:states()
