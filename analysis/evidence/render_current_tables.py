#!/usr/bin/env python3
"""Render or verify final-method tables from the portable current exports."""
from pathlib import Path
import argparse
import pandas as pd
from table_emphasis import format_table
HERE=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--check-only',action='store_true')
p.add_argument('--output-dir',type=Path,help='Write regenerated tables to a separate directory')
args=p.parse_args();check=args.check_only
output=args.output_dir or HERE/'tables'
if not check:output.mkdir(parents=True,exist_ok=True)
s=pd.read_csv(HERE/'data/current/summary.csv').set_index('score')
t=pd.read_csv(HERE/'data/current/task_metrics.csv');done=[]
def emit(file,align,header,rows,group_header=()):
 text='\n'.join([r'\begin{tabular}{@{}'+align+'@{}}',r'\toprule',*group_header,header+r' \\',r'\midrule',*rows,r'\bottomrule',r'\end{tabular}'])+'\n'
 path=output/file
 text=format_table(file,text)
 if check:assert path.read_text()==text,file
 else:path.write_text(text)
 done.append(file)
names={'Ours_OPT':'Ours (OPT)','Ours_Falcon':'Ours (Falcon pair)','AdaDetectGPT':r'AdaDetectGPT$^\dagger$','Binoculars':r'Binoculars$^\dagger$'}
order=['Likelihood','Entropy','Rank','Log-Rank','Fast-DetectGPT','AdaDetectGPT','Binoculars','NTS','Ours_OPT','Ours_Falcon'];rows=[]
for k in order:
 if k=='Ours_OPT':rows.append(r'\midrule')
 r=s.loc[k];vals=[f'{r.auc:.4f}',f'{r.cvar10:.4f}',str(int(r.flips)),f'{100*r.roc_tpr_a01:.2f}',f'{100*r.roc_tpr_a05:.2f}']
 rows.append(names.get(k,k)+' & '+' & '.join(vals)+r' \\')
emit('current_main.tex','lrrrrr',r'Detector & AUROC & CVaR$_{10}$ & Flip & ROC$_1$ & ROC$_5$',rows)
order=['unwatermarked','KGW','SWEET','Unigram','EXP','SynthID','DIP','ITSEdit','CGW','HeavyWater','SemStamp']
source=pd.read_csv(HERE/'data/upper_all_conditions/task_metrics.csv')
values=source.groupby(['watermark_method','score']).auc.mean().unstack()
methods=['Fast-DetectGPT','AdaDetectGPT','NTS','Upper_OPT','Ours_OPT','Upper_Falcon','Ours_Falcon']
rows=[]
for group in order:
 label='Unwatermarked' if group=='unwatermarked' else group
 rows.append(label+' & '+' & '.join(f'{values.loc[group,k]:.4f}' for k in methods)+r' \\')
emit('current_mechanisms.tex','lrrrrrrr',r'Mechanism & FD & Ada$^\dagger$ & NTS & Upper & Two-sided & Upper & Two-sided',rows,
     group_header=[r' & \multicolumn{3}{c}{Baselines} & \multicolumn{2}{c}{Ours (OPT)} & \multicolumn{2}{c}{Ours (Falcon pair)} \\',r'\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(l){7-8}'])
# Tables 2 and 10 include every detector in the final manuscript.
detectors=['Likelihood','Entropy','Rank','Log-Rank','Fast-DetectGPT','AdaDetectGPT','Binoculars','NTS','Ours_OPT','Ours_Falcon']
for field,name,groups,labels in [
 ('generator','current_generators.tex',
  ['GLM-4-9B-0414','falcon','llama-2-7b','opt-2.7B','opt-6.7B'],
  ['GLM-4-9B(0414)','Falcon-7B','Llama-2-7B','OPT-2.7B','OPT-6.7B']),
 ('dataset','current_datasets.tex',
  ['cmv','cnn','dialogsum','eli5','hswag','imdb','pubmed','roct','sci','tldr','wp','yelp'],
  ['CMV','CNN/DM','DialogSum','ELI5','HellaSwag','IMDb','PubMedQA','ROCStories','SciXGen','TLDR News','WritingPrompts','Yelp']),
]:
 values=t.groupby(['score',field]).auc.mean().unstack().loc[detectors,groups]
 rows=[]
 for detector in detectors:
  if detector=='Ours_OPT':rows.append(r'\midrule')
  rows.append(names.get(detector,detector)+' & '+' & '.join(f'{v:.4f}' for v in values.loc[detector])+r' \\')
 emit(name,'l'+'r'*len(groups),'Detector & '+' & '.join(labels),rows)
# Table 7 shares Figure 1's medians and the original-direction OPT task scores.
quantiles=pd.read_csv(HERE/'data/current/score_distributions.csv').set_index('method')
fd=pd.read_csv(HERE/'data/two_sided_controls/fd/by_task.csv')
fd=fd[(fd.proxy=='OPT')&(fd.method=='FD_original_direction')]
rows=[f"Human & {quantiles.loc['human','median']:.3f}"+r' & -- & -- \\']
for method in order:
 block=fd[fd.watermark_method==method]
 assert len(block)==60,method
 label='Unwatermarked' if method=='unwatermarked' else method
 rows.append(f"{label} & {quantiles.loc[method,'median']:.3f} & {block.auc.mean():.4f} & {int((block.auc<.5).sum())}/60"+r' \\')
emit('watermark_mechanism_observations.tex','lrrr',r'Mechanism & Median $S_-$ & Macro AUROC & SOR',rows)
# Table 5: hold the lower branch fixed across the staged upper-branch changes.
fd_summary=pd.read_csv(HERE/'data/two_sided_controls/fd/summary.csv')
shaping=pd.read_csv(HERE/'data/reference_calibration/summary.csv')
human=pd.read_csv(HERE/'data/reference_calibration/human_transitions.csv')
rows=[]
for reference,weight,endpoint in [('Original','Uniform',None),('Original','Entropy','original'),('Shaped','Entropy','shaped')]:
 vals=[]
 for proxy in ['OPT','Falcon']:
  if endpoint is None:
   record=fd_summary[(fd_summary.proxy==proxy)&(fd_summary.method=='FD_same_score_two_tails')].iloc[0]
   auc,power,fpr=record.auc,record.power_a01,record.human_fpr_a01
  else:
   record=shaping[(shaping.proxy==proxy)&(shaping.cohort=='all')].iloc[0]
   h=human[(human.proxy==proxy)&(human.alpha==.01)].iloc[0]
   auc,power,fpr=record['joint_'+endpoint],record['power_'+endpoint+'_a01'],h[endpoint+'_fp']/h.humans
  vals.extend([f'{auc:.4f}',f'{100*power:.2f}',f'{100*fpr:.2f}'])
 rows.append(reference+' & '+weight+' & '+' & '.join(vals)+r' \\')
emit('component_ablation.tex','llrrrrrr',r'Upper reference & Weights & AUROC & Power$_1$ & FPR$_1$ & AUROC & Power$_1$ & FPR$_1$',rows,
     group_header=[r' & & \multicolumn{3}{c}{OPT} & \multicolumn{3}{c}{Falcon pair} \\',r'\cmidrule(lr){3-5}\cmidrule(l){6-8}'])
b=pd.read_csv(HERE/'data/mechanism_evidence/summary.csv');rows=[]
for proxy in ['OPT','Falcon']:
 for wm in ['CGW','HeavyWater']:
  r=b[(b.proxy==proxy)&(b.watermark_method==wm)].iloc[0]
  rows.append(proxy+' & '+wm+' & '+' & '.join(f'{r[k]:.4f}' for k in ['lower_auc','uniform_minus_entropy_gap','center_loss'])+r' \\')
emit('negative_evidence.tex','llrrr',r'Proxy & Mechanism & Lower AUROC & $\Delta_{U-E}$ & $\mathcal L_{\rm center}$',rows)
upper=pd.read_csv(HERE/'data/upper_unwatermarked/summary.csv').set_index('score');rows=[]
for k in ['Fast-DetectGPT','AdaDetectGPT','NTS']:
 rows.append(names.get(k,k)+' & '+f'{upper.loc[k,"auc"]:.4f}'+r' & \textemdash & \textemdash \\')
rows.append(r'\midrule')
for proxy in ['OPT','Falcon']:
 rows.append(names['Ours_'+proxy]+r' & \textemdash & '+f'{upper.loc["Upper_"+proxy,"auc"]:.4f} & {upper.loc["Ours_"+proxy,"auc"]:.4f}'+r' \\')
emit('upper_unwatermarked.tex','lrrr',r'Detector & Baseline & Upper & Complete',rows)
a=pd.read_csv(HERE/'data/calibration/allocation.csv');rows=[]
for proxy in ['OPT','Falcon']:
 for r in a[a.proxy==proxy].to_dict('records'):
  vals=[proxy,f"{r['beta']:.2f}",f"{r['auc']:.4f}",f"{r['unwm_auc']:.4f}"]+[f"{100*r[k]:.2f}" for k in ['power_a01','fpr_a01','power_a05','fpr_a05']]
  rows.append(' & '.join(vals)+r' \\')
emit('review_allocation.tex','lrrrrrrr',r'Proxy & $\beta$ & AUROC & Unwm. & Power$_1$ & FPR$_1$ & Power$_5$ & FPR$_5$',rows)
b=pd.read_csv(HERE/'data/calibration/bank_summary.csv');rows=[]
for proxy in ['OPT','Falcon']:
 for r in b[b.proxy==proxy].to_dict('records'):
  vals=[]
  for k in ['power_a01','fpr_a01']:
   value=f"{100*r[k+'_median']:.2f}"
   if r['replicates']>1:value+=f" [{100*r[k+'_min']:.2f}, {100*r[k+'_max']:.2f}]"
   vals.append(value)
  rows.append(f"{proxy} & {r['m']} & "+' & '.join(vals)+r' \\')
emit('review_bank.tex','lrrr',r'Proxy & $m$ & Power$_1$ & FPR$_1$',rows)
print('PASS:',len(done),'current-detector tables',('verified' if check else 'rendered'))
