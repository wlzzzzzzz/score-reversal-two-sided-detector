"""Summarize all real main-benchmark reference supports and denominator floors."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'scratch/baseline_audits'


def main():
    summaries=[]
    for pair,directory in [('OPT','opt27_test_factorial'),('Falcon','falcon_pair')]:
        records=[]
        for path in sorted((OUT/directory).glob('*.jsonl')):
            for line in path.open():
                r=json.loads(line)
                if r['split']=='test':records.append(r)
        frame=pd.DataFrame(records);assert len(frame)==141660
        human=frame[frame.label.eq(0)].sort_values('generator').drop_duplicates('source_id')
        machine=frame[frame.label.eq(1)]
        groups=[('ALL_human_unique',human),('ALL_machine',machine)]
        groups += [('human_domain_'+str(d),g) for d,g in human.groupby('dataset')]
        groups += [('machine_method_'+str(m),g) for m,g in machine.groupby('watermark_method')]
        for name,g in groups:
            std=g.conditional_std.to_numpy();score=abs(g['T1p2_P0p8_Wentropy'].to_numpy())
            row={'pair':pair,'group':name,'rows':len(g),'epsilon_active_rows':int(g.epsilon_active.sum()),
                 'minimum_conditional_std':float(std.min()),'conditional_std_p01':float(np.quantile(std,.01)),
                 'documents_with_singleton_reference_position':int(g.min_support_tokens.eq(1).sum()),
                 'documents_with_nearzero_reference_variance_position':int((g.nucleus_degenerate_fraction>0).sum()),
                 'document_mean_reference_variance_at_most_1e_minus9_fraction':float(g.nucleus_degenerate_fraction.mean()),
                 'document_mean_observed_outside_reference_fraction':float(g.observed_outside_nucleus_fraction.mean()),
                 'mean_retained_mass':float(g.retained_mass_mean.mean()),
                 'minimum_support_tokens':int(g.min_support_tokens.min()),
                 'mean_effective_weight_tokens':float(g.effective_weight_tokens.mean()),
                 'position_statistic_definition':'nucleus_degenerate_fraction in the scorer counts clamped FP32 conditional variance <=1e-9, not necessarily singleton support'}
            for epsilon in [1e-6,1e-4]:
                changed=score*np.maximum(std,1e-9)/np.maximum(std,epsilon)
                row['changed_rows_epsilon_'+str(epsilon)]=int((std<epsilon).sum())
                row['max_abs_score_change_epsilon_'+str(epsilon)]=float(abs(changed-score).max())
            summaries.append(row)
    (OUT/'analysis').mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(OUT/'analysis/reference_numerical_diagnostics.csv',index=False)
    print(pd.DataFrame(summaries).query("group in ['ALL_human_unique','ALL_machine']").to_string(index=False))


if __name__=='__main__':main()
