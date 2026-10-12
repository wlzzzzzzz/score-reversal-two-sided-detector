"""Check main Human source IDs for exact/near duplicate text across all domains."""
import collections
import itertools
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'scratch/baseline_audits'


def main():
    manifest=json.loads((ROOT/'results/baselines/benchmark/source_manifest.json').read_text())
    rows=[];seen=set()
    for cell in sorted(manifest['cells'],key=lambda c:c['cell']):
        if cell['dataset'] in seen:continue
        seen.add(cell['dataset']);data=json.loads(Path(cell['data_path']).read_text())
        for i,text in enumerate(data['original']):rows.append({'source_id':cell['dataset']+':'+str(i),'dataset':cell['dataset'],'split':cell['split'],'words':re.findall(r'\w+',text.lower())})
    assert len(rows)==3161
    exact=collections.defaultdict(list);postings=collections.defaultdict(list);sizes=[]
    for i,r in enumerate(rows):
        words=r['words'];exact[tuple(words)].append(i);shingles={tuple(words[k:k+5]) for k in range(len(words)-4)};sizes.append(len(shingles))
        for sh in shingles:postings[sh].append(i)
    shared=collections.Counter()
    for ids in postings.values():
        if len(ids)>1:shared.update(itertools.combinations(ids,2))
    near=[]
    for (i,j),n in shared.items():
        containment=n/max(1,min(sizes[i],sizes[j]))
        if containment>=.8:near.append({'left':rows[i]['source_id'],'right':rows[j]['source_id'],'left_split':rows[i]['split'],'right_split':rows[j]['split'],'containment':containment})
    report={'status':'COMPLETE','human_source_ids':len(rows),'unique_normalized_texts':len(exact),
            'normalized_duplicate_groups':[[rows[i]['source_id'] for i in ids] for ids in exact.values() if len(ids)>1],
            'all_shared_5gram_candidates':len(shared),'pairs_containment_ge_0p8':near,
            'scope':'Exhaustive shared5-word-shingle candidates across all main Human sources; no author-independence or semantic-duplicate guarantee.'}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'main_human_source_overlap_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
