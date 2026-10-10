#!/usr/bin/env python3
"""Replay upper-only rank AUROC for all660 tasks using the frozen response cache."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE=Path(__file__).resolve().parents[1]
DEST=HERE/'data/upper_all_conditions'
KEY=['dataset','generator','watermark_method']
ORDER=['Fast-DetectGPT','AdaDetectGPT','NTS','Upper_OPT','Ours_OPT','Upper_Falcon','Ours_Falcon']

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def compute():
    original=pd.read_csv(HERE/'data/current/task_metrics.csv')
    old_upper=pd.read_csv(HERE/'data/upper_unwatermarked/task_metrics.csv')
    sources={s:sha(HERE/s) for s in ['data/current/task_metrics.csv','data/upper_unwatermarked/task_metrics.csv']}
    rows=[];diagnostics=[]
    for proxy in ['OPT','Falcon']:
        path=HERE/'data/sensitivity'/f'cached_{proxy}.npz';sources[str(path.relative_to(HERE))]=sha(path)
        with np.load(path) as z:
            # Same frozen (T=1.2,c=.8,kappa=1.5) column as the existing upper-only export.
            cal=z['cal_upper'][:,2];query=z['query_upper'][:,2]
            assert cal.shape==(800,) and query.shape==(141600,)
            p=(1+len(cal)-np.searchsorted(np.sort(cal),query,side='left'))/801
            lower=(1+np.searchsorted(np.sort(z['cal_lower']),z['query_lower'],side='right'))/801
            upper=-np.log(p);joint=-np.log(np.minimum(p/.5,lower/.5))
            q=pd.DataFrame({k:z[k] for k in ['row_id','source_id',*KEY,'label']})
        assert np.isfinite(upper).all() and np.isfinite(joint).all()
        assert len(q[q.label==0].drop_duplicates('source_id'))==2360
        assert len(q[q.label==1])==129800 and 'wp:11' not in set(q.source_id)
        humans={k:g for k,g in q[q.label==0].groupby(['dataset','generator'])}
        complete=original[original.score=='Ours_'+proxy].set_index(KEY)
        previous=old_upper[old_upper.score=='Upper_'+proxy].set_index(KEY)
        count=0
        for (ds,gen,wm),m in q[q.label==1].groupby(KEY,sort=True):
            h=humans[(ds,gen)]
            assert h.source_id.is_unique and m.source_id.is_unique and set(h.source_id)==set(m.source_id)
            ix=np.r_[h.index,m.index];labels=np.r_[np.zeros(len(h)),np.ones(len(m))]
            a=roc_auc_score(labels,upper[ix]);joint_a=roc_auc_score(labels,joint[ix])
            np.testing.assert_allclose(joint_a,complete.loc[(ds,gen,wm),'auc'],atol=1e-12,rtol=0)
            if wm=='unwatermarked':np.testing.assert_allclose(a,previous.loc[(ds,gen,wm),'auc'],atol=1e-12,rtol=0)
            rows.append(dict(score='Upper_'+proxy,dataset=ds,generator=gen,watermark_method=wm,n_human=len(h),n_machine=len(m),auc=a))
            diagnostics.append(dict(proxy=proxy,dataset=ds,generator=gen,watermark_method=wm,raw_upper_auc=roc_auc_score(labels,query[ix]),calibrated_upper_auc=a,complete_auc=joint_a))
            count+=1
        assert count==660
    columns=['score',*KEY,'n_human','n_machine','auc']
    tasks=pd.concat([original[original.score.isin(ORDER)][columns],pd.DataFrame(rows)],ignore_index=True).sort_values(['score',*KEY]).reset_index(drop=True)
    assert len(tasks)==4620 and not tasks.duplicated(['score',*KEY]).any()
    ref=tasks[tasks.score=='Upper_OPT'].set_index(KEY).sort_index()
    for method in ORDER:
        part=tasks[tasks.score==method].set_index(KEY).sort_index();assert len(part)==660
        pd.testing.assert_index_equal(part.index,ref.index)
        np.testing.assert_array_equal(part[['n_human','n_machine']],ref[['n_human','n_machine']])
    summary=tasks.groupby(['score','watermark_method']).agg(tasks=('auc','size'),auc=('auc','mean')).reset_index()
    assert (summary.tasks==60).all()
    protocol=dict(status='PASS',one_sided='Upper only: J_plus=-log(p_plus), inclusive add-one human ranks; fixed direction for every mechanism',two_sided='Current asymmetric J, equal tail allocation beta=.5',T=1.2,c=.8,kappa=1.5,calibration_cells=20,calibration_unique_humans=800,test_cells=60,tasks_per_score=660,test_unique_humans=2360,test_machine_texts=129800,aggregation='Equal-weight mean of60 task AUROCs per mechanism',AdaDetectGPT='Existing original-direction adapted Falcon pair; not the two-tailed Ada control',parameter_selection=False,new_model_forwards=False,full_joint_and_previous_unwatermarked_parity_tolerance=1e-12,source_sha256=sources)
    return {'task_metrics.csv':tasks,'by_mechanism.csv':summary,'rank_diagnostics.csv':pd.DataFrame(diagnostics)},protocol

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--check-only',action='store_true');args=ap.parse_args()
    frames,protocol=compute()
    if args.check_only:
        for name,f in frames.items():pd.testing.assert_frame_equal(pd.read_csv(DEST/name),f,check_exact=False,atol=1e-12,rtol=0)
        assert json.loads((DEST/'PROTOCOL.json').read_text())==protocol
    else:
        DEST.mkdir(parents=True,exist_ok=True)
        for name,f in frames.items():f.to_csv(DEST/name,index=False,float_format='%.17g')
        (DEST/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print('PASS: seven scores,660 matched tasks each; existing complete and unwatermarked upper results reproduced.')
