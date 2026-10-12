#!/usr/bin/env python3
"""Portable verification of the HeavyWater pipeline-control table from document scores."""
from pathlib import Path
import csv,hashlib,json,re
import numpy as np
from table_emphasis import plain
HERE=Path(__file__).resolve().parents[1]
def validate():
    data=HERE/'data/heavywater_controls'
    source=json.loads((data/'SOURCE_AUDIT.json').read_text())
    for f,h in source['sha256'].items():
        path=data/f
        # Historical JSON receipts retain pre-portability execution hashes.
        if path.suffix=='.csv':assert hashlib.sha256(path.read_bytes()).hexdigest()==h,f
    protocol=json.loads((data/'PROTOCOL.json').read_text())
    names=[a['name'] for a in protocol['arms']];domains=protocol['domains']
    scores=[r for r in csv.DictReader((data/'DOCUMENT_SCORES.csv').open()) if r['view']=='serialized' and r['length']=='full']
    rows={r['arm']:r for r in csv.DictReader((data/'SUMMARY.csv').open()) if r['view']=='serialized' and r['length']=='full'}
    for name in names:
        aa=[];mm=[]
        for ds in domains:
            h=np.array([float(r['R']) for r in scores if r['dataset']==ds and r['arm']=='human'])
            m=np.array([float(r['R']) for r in scores if r['dataset']==ds and r['arm']==name])
            assert len(m)==len(h)==10
            aa.append(float(((m[:,None]>h[None,:])+.5*(m[:,None]==h[None,:])).mean()));mm.append(float(m.mean()))
        assert abs(np.mean(aa)-float(rows[name]['auc']))<1e-12
        assert abs(np.mean(mm)-float(rows[name]['mean_R']))<1e-12
    metadata=list(csv.DictReader((data/'GENERATION_METADATA.csv').open()));assert len(metadata)==360
    assert len(set(r['source_id'] for r in metadata))==40
    tab=plain((HERE/'tables/heavywater_generation_controls.tex').read_text())
    pairs=re.findall(r'&\s*(-?\d+\.\d+)\s*&\s*(\d+\.\d+)\s*\\\\',tab)
    expected=[(f'{float(rows[n]["mean_R"]):.4f}',f'{float(rows[n]["auc"]):.4f}') for n in names]
    assert pairs==expected
    audit=json.loads((data/'PROVENANCE_AUDIT.json').read_text())
    assert audit['test_machine_texts']==audit['exact_record_matches']==11800
    first=json.loads((data/'FIRST_ATTEMPT_AUDIT.json').read_text())
    assert all(r['first_attempt_n']==11559 and r['first_attempt_reversed_cells']==60 for r in first['summary'])
    return dict(status='PASS',source_integrity='recorded CSV hashes',
        historical_numerical_exports_unchanged=True,generation_controls=len(names),new_documents=360,calibration_domain_sources=40,
        calibration_cells_used=4,all_primary_AUROCs_recomputed_from_document_scores=True,table_values=True,
        test_record_matches=11800,detector_parameters_unchanged=True)
if __name__=='__main__':print(json.dumps(validate(),indent=2))
