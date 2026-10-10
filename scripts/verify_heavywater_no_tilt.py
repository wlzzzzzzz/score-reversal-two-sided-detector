#!/usr/bin/env python3
"""Small CPU checks of the actual no-tilt HeavyWater transport implementation."""
import contextlib
import hashlib
import importlib
import io
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
import ot

ROOT=Path(__file__).resolve().parents[1]
IMPLEMENTATION=ROOT/'third_party/external_watermarks/HeavyWater_SimplexWater'
SOURCE=IMPLEMENTATION/'watermark/heavy_tail_randscore.py'
DEPENDENCY=IMPLEMENTATION/'watermark/old_watermark.py'
PROTOCOL=ROOT/'docs/protocols/heavywater_numerics.md'
OUT=ROOT/'scratch/heavywater_no_tilt_numerics'
GRID=((16,64),(32,128),(64,1024))
SEEDS=(15485863,20260913)
PROFILES=('uniform','geometric','peaked','singleton_control')
DISTRIBUTIONS=('lognormal','gamma')
LENGTHS=(1,16,256)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def profile(n,name):
    if name=='uniform':return torch.full((n,),1/n,dtype=torch.float32,device='cpu')
    if name=='geometric':return torch.softmax(-.4*torch.arange(n,dtype=torch.float32),dim=0)
    peak=.97 if name=='peaked' else .9995
    out=torch.full((n,),(1-peak)/(n-1),dtype=torch.float32,device='cpu')
    out[0]=peak
    return out


def normalized(array):
    a=np.asarray(array,dtype=np.float64)
    return a/a.sum()


def sequence_statistics(case_id,score_name,Q,ell,baseline):
    means=ell@Q
    variances=np.sum(Q*(ell[:,None]-means[None,:])**2,axis=0)
    mean=float(means.mean());within=float(variances.mean());between=float(np.var(means))
    marginal=Q.mean(axis=1)
    total=float(np.dot(marginal,(ell-mean)**2))
    assert abs(total-within-between)<1e-10
    rows=[]
    for n in LENGTHS:
        fresh=(within+between)/n
        repeated=within/n+between
        assert abs(repeated-fresh-(1-1/n)*between)<1e-12
        assert repeated+1e-12>=fresh
        for name,expected,var in (
            ('fresh_uniform',mean,fresh),
            ('repeated_state1',float(means[0]),float(variances[0])/n),
            ('repeated_uniform_state',mean,repeated)):
            rows.append({'case_id':case_id,'score':score_name,'document_tokens':n,'state_process':name,
                         'document_mean_expectation':expected,'mean_drift_from_processed_input':expected-baseline,
                         'document_mean_variance':var,'within_state_token_variance':within,
                         'between_state_token_mean_variance':between,
                         'reused_minus_fresh_variance':repeated-fresh})
    return rows,float(abs(total-within-between))


def hand_example():
    p=np.array([.9,.1])
    Q=np.array([[.82,.98],[.18,.02]])
    ell=np.log(p)
    np.testing.assert_allclose(Q.mean(axis=1),p,rtol=0,atol=1e-15)
    drift=ell@Q-np.dot(ell,p)
    np.testing.assert_allclose(drift,[-.08*np.log(9),.08*np.log(9)],rtol=0,atol=1e-15)
    rows,error=sequence_statistics('hand_two_token_two_state','aligned_logprob',Q,ell,float(ell@p))
    return {'p':p.tolist(),'Q_by_column':Q.tolist(),'state_drifts':drift.tolist(),
            'uniform_state_mean_drift':float(drift.mean()),'law_total_variance_error':error,
            'sequence_rows':rows,'construction':'valid strictly positive marginal-preserving coupling; not a generated-cost claim'}


def run_case(module,vocab,k,seed,distribution,profile_name):
    case_id=f'V{vocab}_K{k}_{distribution}_seed{seed}_{profile_name}'
    p=profile(vocab,profile_name)
    pnorm=normalized(p.numpy())
    capture_stdout=io.StringIO()
    with contextlib.redirect_stdout(capture_stdout):
        cost=module.generate_normalized_lognormal_cost_matrix(vocab,S_size=k,mean=0.,sigma=1.,seed=seed,device=torch.device('cpu'),dist=distribution)
    instance=module.HeavyTailLogitsProcessor.__new__(module.HeavyTailLogitsProcessor)
    instance.vocab_size=vocab;instance.k=k;instance.top_p=.999;instance.temperature=1.
    instance.sinkhorn_reg=.05;instance.sinkhorn_thresh=1e-5
    instance.tilt=False;instance.tilting_delta=.1;instance.cost_matrix=cost;instance.device=torch.device('cpu')
    instance.hash_key=seed
    keep=module.top_p_indices(p,.999)
    assert p.device.type==cost.device.type=='cpu'
    captured={}
    original_sinkhorn=module.ot.sinkhorn
    def recording_sinkhorn(*args,**kwargs):
        result=original_sinkhorn(*args,**kwargs)
        captured['args']=args;captured['kwargs']=kwargs
        captured['plan']=result.detach().clone()
        return result
    started=time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        module.ot.sinkhorn=recording_sinkhorn
        try:
            actual_first=instance.tilt_q_HT(p.clone(),1,num_iter=2000)
        finally:
            module.ot.sinkhorn=original_sinkhorn
        solver_called=bool(captured)
        log_info={}
        if solver_called:
            P=captured['plan']
            logged_plan,log_info=original_sinkhorn(*captured['args'],**captured['kwargs'],log=True)
            solver_log_parity=float(torch.max(torch.abs(logged_plan-P)).item())
            assert solver_log_parity<=1e-7
            assert torch.isfinite(P).all() and bool((P.sum(dim=0)>0).all())
            P64=P.double().numpy()
            a_raw=captured['kwargs']['a'].double().numpy()
            col_target=captured['kwargs']['b'].double().numpy()
            row_l1=float(np.abs(P64.sum(axis=1)-a_raw).sum())
            col_l1=float(np.abs(P64.sum(axis=0)-col_target).sum())
            Qsmall=P64/P64.sum(axis=0,keepdims=True)
            Q=np.zeros((vocab,k));Q[keep.numpy(),:]=Qsmall
            # Reconstruct all states using the same per-column FP32 division as the actual method.
            Q32small=torch.stack([P[:,s]/P[:,s].sum() for s in range(k)],dim=1)
            Q32=np.zeros((vocab,k));Q32[keep.numpy(),:]=Q32small.double().numpy()
            rounding_l1=float(np.abs(Q32-Q).sum(axis=0).mean())
            processed=np.zeros(vocab);processed[keep.numpy()]=normalized(a_raw)
            processed_raw=np.zeros(vocab);processed_raw[keep.numpy()]=a_raw
            target_rounding_l1=float(np.abs(processed-processed_raw).sum())
            marginal=Q.mean(axis=1)
            l1_raw=float(np.abs(marginal-processed_raw).sum())
            l1=float(np.abs(marginal-processed).sum())
            bound=row_l1+col_l1
            assert l1_raw<=bound+1e-12
            assert l1<=bound+target_rounding_l1+1e-12
            # The first primary call plus two fixed states; no state selection by score.
            actuals={1:actual_first}
            for s in (k//2,k):actuals[s]=instance.tilt_q_HT(p.clone(),s,num_iter=2000)
            actual_parity=max(float(np.max(np.abs(value.double().numpy()-Q[:,s-1]))) for s,value in actuals.items())
            actual_fp32_parity=max(float(np.max(np.abs(value.double().numpy()-Q32[:,s-1]))) for s,value in actuals.items())
            assert actual_parity<2e-7 and actual_fp32_parity<2e-7
            last_error=float(log_info['err'][-1]) if log_info.get('err') else None
            iterations=int(log_info.get('niter',-1))
            converged=bool(last_error is not None and last_error<=instance.sinkhorn_thresh)
        else:
            assert len(keep)==1
            np.testing.assert_array_equal(actual_first.numpy(),p.numpy())
            processed=pnorm.copy();processed_raw=p.double().numpy()
            Q=np.repeat(pnorm[:,None],k,axis=1);Q32=np.repeat(p.double().numpy()[:,None],k,axis=1)
            marginal=Q.mean(axis=1)
            row_l1=col_l1=bound=l1_raw=l1=rounding_l1=target_rounding_l1=0.
            actual_parity=actual_fp32_parity=solver_log_parity=0.
            last_error=None;iterations=0;converged=None
            P64=np.empty((0,0))
        warning_strings=sorted(set(str(w.message) for w in caught))
    elapsed=time.perf_counter()-started
    score_rows,sequence_rows=[],[]
    max_variance_error=0.
    scores={'aligned_logprob':np.log(pnorm),
            'mismatched_proxy_logprob':None}
    i=np.arange(vocab,dtype=np.float64)
    logits=np.sin(.7*i)+.4*np.cos(.13*i)
    scores['mismatched_proxy_logprob']=logits-np.log(np.exp(logits).sum())
    for name,ell in scores.items():
        base=float(ell@processed);base_original=float(ell@pnorm)
        state_means=ell@Q;state_drift=state_means-base
        mean=float(state_means.mean());drift=mean-base
        delta=marginal-processed
        score_bound=float(np.max(np.abs(ell))*np.abs(delta).sum())
        midpoint=float((ell.max()+ell.min())/2)
        centered_bound=float((ell.max()-ell.min())/2*np.abs(delta).sum()+abs(midpoint)*abs(delta.sum()))
        assert abs(drift)<=score_bound+1e-12 and abs(drift)<=centered_bound+1e-12
        score_rows.append({'case_id':case_id,'vocabulary':vocab,'states':k,'cost_distribution':distribution,
                           'seed':seed,'profile':profile_name,'score':name,'solver_called':solver_called,
                           'processed_input_expectation':base,'original_input_expectation':base_original,
                           'uniform_state_output_expectation':mean,'uniform_state_drift_processed':drift,
                           'uniform_state_drift_original':mean-base_original,
                           'top_p_only_score_drift':base-base_original,
                           'state1_drift_processed':float(state_drift[0]),
                           'minimum_state_drift':float(state_drift.min()),'maximum_state_drift':float(state_drift.max()),
                           'positive_state_count':int((state_drift>1e-8).sum()),
                           'negative_state_count':int((state_drift < -1e-8).sum()),
                           'near_zero_state_count':int((np.abs(state_drift)<=1e-8).sum()),
                           'score_infinity_norm_bound':score_bound,'centered_range_score_bound':centered_bound})
        seq,verror=sequence_statistics(case_id,name,Q,ell,base)
        sequence_rows.extend(seq);max_variance_error=max(max_variance_error,verror)
    case={'case_id':case_id,'vocabulary':vocab,'states':k,'seed':seed,'cost_distribution':distribution,
          'profile':profile_name,'tilt':False,'temperature':1.,'top_p':.999,'reg':.05,'stop_threshold':1e-5,
          'support_size':len(keep),'retained_original_mass':float(p[keep].sum().item()),
          'singleton_bypass':not solver_called,'solver_called':solver_called,'solver_converged':converged,
          'solver_iterations':iterations,'last_logged_error':last_error,'row_l1_residual':row_l1,
          'column_l1_residual':col_l1,'row_plus_column_bound':bound,
          'uniform_output_vs_processed_raw_l1':l1_raw,'uniform_output_vs_processed_normalized_l1':l1,
          'target_normalization_rounding_l1':target_rounding_l1,'fp32_column_normalization_mean_l1_error':rounding_l1,
          'actual_method_reconstruction_max_error':actual_parity,'actual_method_fp32_reconstruction_max_error':actual_fp32_parity,
          'solver_log_plan_max_error':solver_log_parity,
          'uniform_output_vs_original_input_l1':float(np.abs(marginal-pnorm).sum()),
          'cost_max_absolute_row_mean':float(np.max(np.abs(cost.double().numpy().mean(axis=1)))),
          'cost_max_row_std_error':float(np.max(np.abs(cost.double().numpy().std(axis=1)-1))),
          'law_total_variance_max_error':max_variance_error,'warnings':'; '.join(warning_strings),
          'cost_generator_stdout':capture_stdout.getvalue().strip(),'cpu_seconds':elapsed}
    np.savez_compressed(OUT/'matrices'/f'{case_id}.npz',input_probability=pnorm,processed_probability=processed,
                        cost=cost.numpy(),transport_plan=P64,conditional_probabilities=Q,
                        conditional_probabilities_fp32=Q32,retained_indices=keep.numpy())
    return case,score_rows,sequence_rows


def markdown(df):
    rows=['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']
    for r in df.itertuples(index=False,name=None):
        rows.append('| '+' | '.join(f'{x:.7g}' if isinstance(x,(float,np.floating)) else str(x) for x in r)+' |')
    return '\n'.join(rows)


def main():
    torch.set_num_threads(1)
    sys.path.insert(0,str(IMPLEMENTATION))
    module=importlib.import_module('watermark.heavy_tail_randscore')
    assert Path(module.__file__).resolve()==SOURCE.resolve()
    assert Path(importlib.import_module('watermark.old_watermark').__file__).resolve()==DEPENDENCY.resolve()
    OUT.mkdir(parents=True,exist_ok=False);(OUT/'matrices').mkdir()
    frozen={'protocol_sha256':sha(PROTOCOL),'script_sha256':sha(__file__),
            'implementation_sha256':sha(SOURCE),'dependency_sha256':sha(DEPENDENCY),
            'torch_version':torch.__version__,'pot_version':ot.__version__,'numpy_version':np.__version__,
            'device':'cpu','torch_threads':torch.get_num_threads(),'new_LM_forward':False,
            'grid':GRID,'seeds':SEEDS,'profiles':PROFILES,'distributions':DISTRIBUTIONS,
            'tilt':False,'top_p':.999,'temperature':1.,'reg':.05,'threshold':1e-5,'max_iterations':2000}
    dump(OUT/'FROZEN.json',frozen)
    hand=hand_example();dump(OUT/'HAND_COUNTEREXAMPLE.json',hand)
    cases,scores,sequences=[],[],[]
    for vocab,k in GRID:
        for distribution in DISTRIBUTIONS:
            for seed in SEEDS:
                for pname in PROFILES:
                    case,sc,seq=run_case(module,vocab,k,seed,distribution,pname)
                    cases.append(case);scores.extend(sc);sequences.extend(seq)
        print(f'Finished vocabulary={vocab}, states={k}: {len(cases)} fixed cases',flush=True)
    c,s,z=pd.DataFrame(cases),pd.DataFrame(scores),pd.DataFrame(sequences)
    c.to_csv(OUT/'cases.csv',index=False);s.to_csv(OUT/'score_drift.csv',index=False);z.to_csv(OUT/'sequence_moments.csv',index=False)
    solver=c[c.solver_called]
    score_solver=s[s.solver_called]
    overview=solver.groupby(['states','cost_distribution']).agg(
        cases=('case_id','size'),nonconverged=('solver_converged',lambda x:int((~x.astype(bool)).sum())),
        maximum_row_L1=('row_l1_residual','max'),maximum_column_L1=('column_l1_residual','max'),
        maximum_marginal_L1=('uniform_output_vs_processed_normalized_l1','max'),
        maximum_actual_parity_error=('actual_method_reconstruction_max_error','max')).reset_index()
    overview.to_csv(OUT/'solver_summary.csv',index=False)
    selected=s[s.profile.eq('geometric')][['states','cost_distribution','seed','score','uniform_state_drift_processed',
                                        'uniform_state_drift_original','state1_drift_processed','minimum_state_drift','maximum_state_drift']]
    diag={**frozen,'cases':len(c),'solver_cases':len(solver),'singleton_bypass_cases':int(c.singleton_bypass.sum()),
          'nonconverged_cases':int((~solver.solver_converged.astype(bool)).sum()),
          'maximum_marginal_L1':float(solver.uniform_output_vs_processed_normalized_l1.max()),
          'maximum_absolute_uniform_state_score_drift':float(score_solver.uniform_state_drift_processed.abs().max()),
          'maximum_actual_method_reconstruction_error':float(c.actual_method_reconstruction_max_error.max()),
          'maximum_solver_log_plan_error':float(c.solver_log_plan_max_error.max()),
          'maximum_law_total_variance_error':float(c.law_total_variance_max_error.max()),
          'marginal_residual_bound_all_passed':True,'score_drift_bounds_all_passed':True,
          'state_reuse_variance_identity_all_passed':True,'hand_counterexample_passed':True,
          'solver_warnings_cases':int(c.warnings.ne('').sum()),'cpu_solver_seconds_sum':float(c.cpu_seconds.sum())}
    dump(OUT/'FINAL_AUDIT.json',diag)
    report = [
        '# HeavyWater without tilting: CPU numerical checks',
        ('The fixed grid calls the actual cost generator, POT Sinkhorn solver and '
         '`HeavyTailLogitsProcessor.tilt_q_HT` on CPU. No language model is loaded. '
         'It includes 48 cases: 36 transport solves and 12 singleton bypasses.'),
        markdown(overview),
        (f'Maximum marginal L1 residual: {diag["maximum_marginal_L1"]:.8g}. '
         f'Maximum absolute uniform-state score drift: '
         f'{diag["maximum_absolute_uniform_state_score_drift"]:.8g}. '
         f'{diag["nonconverged_cases"]} solver cases did not converge. '
         'Solver warnings and residuals are retained.'),
        '## Fixed-profile results', markdown(selected),
        ('Score drift is measured against both the original distribution and the '
         'input after top-p processing. Full results are in `score_drift.csv`.'),
        '## State reuse',
        ('Repeated uniformly selected states add `(1 - 1/n) Var_s(mu_s)` to the '
         'variance of the sequence mean relative to fresh independent states. '
         'The checks use sequence lengths 1, 16 and 256.'),
        '## Two-state counterexample',
        ('For `p=(0.9, 0.1)`, `Q1=(0.82, 0.18)` and `Q2=(0.98, 0.02)`, the '
         'uniform-state marginal equals p, while fixed states have log-probability '
         'drifts `-0.08 log(9)` and `+0.08 log(9)`. This is an analytic coupling, '
         'not a claim about a generated cost matrix.'),
        '## Audit scope',
        ('`FROZEN.json` and `FINAL_AUDIT.json` record protocol and implementation '
         'hashes, environment details and numerical diagnostics. `matrices/` retains the transport arrays. These synthetic '
         'checks examine fixed-state drift and solver residuals; they do not '
         'establish detector power on language-model outputs.'),
    ]
    (OUT/'REPORT.md').write_text('\n\n'.join(report))
    print(overview.to_string(index=False),flush=True)
    print(json.dumps({k:v for k,v in diag.items() if k not in frozen},indent=2),flush=True)


if __name__=='__main__':main()
