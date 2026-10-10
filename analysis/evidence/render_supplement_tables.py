#!/usr/bin/env python3
"""Render tables of the current test's attributed decisions and sensitivity."""
from pathlib import Path
import argparse
import pandas as pd
from table_emphasis import format_table
HERE=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--check-only',action='store_true');check=p.parse_args().check_only
done=[]
def emit(name,align,header,rows):
    text='\n'.join([r'\begin{tabular}{@{}'+align+'@{}}',r'\toprule',header+r' \\',r'\midrule',*rows,r'\bottomrule',r'\end{tabular}'])+'\n'
    path=HERE/'tables'/name
    text=format_table(name,text)
    if check:assert path.read_text()==text,name
    else:path.write_text(text)
    done.append(name)
def pct(v):return f'{100*v:.2f}'
def pct_precise(v):return (r'$<0.01$' if 0<v<.0001 else pct(v))
def label(w):return {'unwatermarked':'Unwatermarked','Human':'Human (FPR)'}.get(w,w)

d=pd.read_csv(HERE/'data/directional/attribution.csv').set_index(['proxy','alpha','watermark_method'])
order=['unwatermarked','KGW','SWEET','Unigram','EXP','SynthID','DIP','ITSEdit','CGW','HeavyWater','SemStamp','All machine','Human']
for alpha,tag in [(.01,'a01')]:
    rows=[r' & \multicolumn{4}{c}{OPT} & \multicolumn{4}{c}{Falcon pair} \\',r'\cmidrule(lr){2-5}\cmidrule(l){6-9}',r'Population & Upper & Lower & Both & Union & Upper & Lower & Both & Union \\',r'\midrule']
    for wm in order:
        if wm=='All machine':rows.append(r'\midrule')
        vals=[pct_precise(d.loc[(proxy,alpha,wm),k]) for proxy in ['OPT','Falcon'] for k in ['upper_only','lower_only','both','union']]
        rows.append(label(wm)+' & '+' & '.join(vals)+r' \\')
    # Empty header is suppressed to keep the two-level heading compact.
    emit('tail_attribution_'+tag+'.tex','lrrrrrrrr',r'\multicolumn{9}{c}{Nominal total budget '+str(int(alpha*100))+r'\%}',rows)

d=pd.read_csv(HERE/'data/sensitivity/temperature_mass.csv');rows=[]
for proxy in ['OPT','Falcon']:
    for r in d[d.proxy==proxy].to_dict('records'):
        vals=[proxy,f"{r['T']:.1f}",f"{r['c']:.1f}",f"{r['auc']:.4f}",f"{r['cvar10']:.4f}"]+[pct(r[k]) for k in ['power_a01','fpr_a01','power_a05','fpr_a05']]
        rows.append(' & '.join(vals)+r' \\')
    if proxy=='OPT':rows.append(r'\midrule')
emit('temperature_mass.tex','lrrrrrrrr',r'Proxy & $T$ & $c$ & AUROC & CVaR$_{10}$ & Power$_1$ & FPR$_1$ & Power$_5$ & FPR$_5$',rows)

d=pd.read_csv(HERE/'data/sensitivity/kappa.csv');rows=[];effects=[]
for r in d.to_dict('records'):
    vals=[f"{r['kappa']:g}",f"{r['auc']:.4f}",f"{r['cvar10']:.4f}"]+[pct(r[k]) for k in ['power_a01','fpr_a01','power_a05','fpr_a05']]+[f"{r['human_neff_ratio_median']:.3f}"]
    rows.append(' & '.join(vals)+r' \\')
    delta=f"{r['auc_delta_vs_default']:+.4f} [{r['auc_delta_low']:+.4f}, {r['auc_delta_high']:+.4f}]"
    if r['kappa']==1.5:delta='Reference'
    effects.append(f"{r['kappa']:g} & {delta} & "+' & '.join(pct(r[k]) for k in ['human_cap_fraction_median','machine_cap_fraction_median'])+r' \\')
emit('kappa_sensitivity.tex','rrrrrrrr',r'$\kappa$ & AUROC & CVaR$_{10}$ & Power$_1$ & FPR$_1$ & Power$_5$ & FPR$_5$ & $n_{\rm eff}/n$ (H)',rows)
emit('kappa_effects.tex','rlrr',r'$\kappa$ & $\Delta$AUROC [95\% interval] & Capped positions (H) & Capped positions (M)',effects)

d=pd.read_csv(HERE/'data/length_strata/summary.csv');perf=[];cal=[]
for proxy in ['OPT','Falcon']:
    perf.append(r'\multicolumn{7}{c}{'+proxy+r'} \\');cal.append(r'\multicolumn{6}{c}{'+proxy+r'} \\')
    for r in d[d.proxy==proxy].to_dict('records'):
        interval=f"{r['lower_exclusive']+1}--{r['upper_inclusive']}"
        vals=[interval,str(r['n_human']),str(r['n_machine']),str(r['supported_tasks']),f"{r['macro_auc']:.4f}",pct(r['power_a01']),pct(r['power_a05'])]
        perf.append(' & '.join(vals)+r' \\')
        ci=lambda tag:f"{pct(r['fpr_'+tag])} [{pct(r['fpr_'+tag+'_low'])}, {pct(r['fpr_'+tag+'_high'])}]"
        vals=[interval,f"{r['fp_a01']}/{r['n_human']}",ci('a01'),f"{r['fp_a05']}/{r['n_human']}",ci('a05'),f"{r['human_neff_ratio_median']:.3f}"]
        cal.append(' & '.join(vals)+r' \\')
    if proxy=='OPT':perf.append(r'\midrule');cal.append(r'\midrule')
emit('length_performance.tex','lrrrrrr',r'Positions $n$ & Humans & Machines & \shortstack{Evaluation\\settings} & AUROC & Power$_1$ & Power$_5$',perf)
emit('length_calibration.tex','lrlrlr',r'Positions $n$ & FP$_1$/H & FPR$_1$ [95\%] & FP$_5$/H & FPR$_5$ [95\%] & $n_{\rm eff}/n$',cal)

path=HERE/'data/sensitivity/local_construction.csv'
if path.exists():
    rows=[]
    for r in pd.read_csv(path).to_dict('records'):
        vals=[f"{r['T']:.1f}",f"{r['c']:.1f}",f"{r['auc']:.4f}",f"{r['cvar10']:.4f}"]+[pct(r[k]) for k in ['power_a01','fpr_a01','power_a05','fpr_a05']]
        rows.append(' & '.join(vals)+r' \\')
    emit('local_construction.tex','rrrrrrrr',r'$T$ & $c$ & AUROC & CVaR$_{10}$ & Power$_1$ & FPR$_1$ & Power$_5$ & FPR$_5$',rows)
print('PASS:',len(done),'supplement tables',('verified' if check else 'rendered'))
