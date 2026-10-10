"""Separate FP16 forward batch dependence from conditional-moment errors."""
import hashlib
import json
from pathlib import Path
import torch
from model import load_model,load_tokenizer
from audit_iclr_numerics import s3_only
from io_utils import atomic_json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'scratch/iclr_review_supplement'


def main():
    old={}
    for path in (ROOT/'scratch/model_pair_s3_full_opt27_self/precompute').glob('shard_*.jsonl'):
        for line in path.open():
            r=json.loads(line);old[r['row_id']]=r['nucleus_entropy']
    new={}
    for path in (OUT/'opt27_test_factorial').glob('*.jsonl'):
        if '.partial.' in path.name:continue
        for line in path.open():
            r=json.loads(line);new[r['row_id']]=r['T1p2_P0p8_Wentropy']
    ranked=sorted(new,key=lambda k:abs(new[k]-old[k]),reverse=True)
    fixed=sorted(new,key=lambda k:hashlib.sha256(('20260907:'+k).encode()).hexdigest())
    chosen=list(dict.fromkeys(ranked[:16]+fixed[:16]))
    manifest=json.loads((ROOT/'results/baselines/benchmark/source_manifest.json').read_text())
    cfg=json.loads((ROOT/'configs/original/opt_factorial.json').read_text())
    tok=load_tokenizer('models/opt-2.7B','models');groups={};target_groups={}
    for cell in manifest['cells']:
        names=[k for k in chosen if k.startswith(cell['dataset']+':'+cell['generator']+':')]
        if not names:continue
        data=json.loads(Path(cell['data_path']).read_text());records=[]
        for method in ['human']+cfg['methods']:
            for i,text in enumerate(data['original' if method=='human' else method]):
                ids=tok(text,return_token_type_ids=False)['input_ids'][:2048]
                records.append((ids,f"{cell['dataset']}:{cell['generator']}:{i}:{method}"))
        records.sort(key=lambda r:len(r[0]));offset=0
        while offset<len(records):
            stop=min(offset+8,len(records))
            while stop>offset+1 and len(records[stop-1][0])*(stop-offset)>2048:stop-=1
            batch=records[offset:stop]
            for i,(ids,name) in enumerate(batch):
                if name in names:
                    key=cell['cell']+':'+str(offset);groups[key]=batch;target_groups[name]=(key,i)
            offset=stop
    model=load_model('models/opt-2.7B','cuda:0','models').eval()
    result={name:{'row_id':name,'legacy_batch1_score':old[name],'new_batch_score':new[name],
                  'selected_for':'largest_observed_numeric_difference' if name in ranked[:16] else 'fixed_hash_sample'} for name in chosen}
    for precision in ['float16','float32']:
        if precision=='float32':model=model.float()
        for key,batch in groups.items():
            encoded=tok.pad([{'input_ids':ids,'attention_mask':[1]*len(ids)} for ids,_ in batch],return_tensors='pt').to('cuda:0')
            with torch.inference_mode():logits=model(**encoded,use_cache=False).logits
            for i,(ids,name) in enumerate(batch):
                if name not in result:continue
                n=len(ids)-1;r=logits[i,:n];lab=encoded.input_ids[i,1:n+1]
                result[name][precision+'_batched']=s3_only(r,r,lab)['score'];result[name]['batch_size']=len(batch);result[name]['input_tokens']=len(ids)
                if precision=='float16':result[name]['float16_batched_fp64_moments']=s3_only(r,r,lab,dtype=torch.float64,stable_variance=True)['score']
            del logits
        for name in chosen:
            key,i=target_groups[name];ids=groups[key][i][0];encoded=torch.tensor([ids],device='cuda:0')
            with torch.inference_mode():logits=model(input_ids=encoded,attention_mask=torch.ones_like(encoded),use_cache=False).logits
            r=logits[0,:-1];lab=encoded[0,1:];result[name][precision+'_singleton']=s3_only(r,r,lab)['score']
            del logits,r
        print(json.dumps({'precision':precision,'texts':len(result)}),flush=True)
    records=list(result.values())
    legacy_error=max(abs(r['float16_singleton']-r['legacy_batch1_score']) for r in records)
    batch_error=max(abs(r['float16_batched']-r['new_batch_score']) for r in records)
    audit={'status':'COMPLETE','selection':'16 largest cache differences plus16 hash-selected texts, for numerical diagnosis only',
           'available_completed_rows_when_selected':len(new),'cases':records,'legacy_singleton_replay_max_error':legacy_error,
           'new_batch_replay_max_error':batch_error,
           'max_FP16_batch_vs_singleton_difference':max(abs(r['float16_batched']-r['float16_singleton']) for r in records),
           'max_FP32_batch_vs_singleton_difference':max(abs(r['float32_batched']-r['float32_singleton']) for r in records),
           'max_same_logits_fp64_moment_change':max(abs(r['float16_batched']-r['float16_batched_fp64_moments']) for r in records),
           'checkpoint':'Same original FP16 checkpoint weights; model.float() for the FP32 forward sensitivity only.'}
    atomic_json(OUT/'batch_precision_audit.json',audit,sort_keys=False)
    assert legacy_error<1e-4 and batch_error<1e-4,audit
    print(json.dumps({k:v for k,v in audit.items() if k!='cases'}),flush=True)


if __name__=='__main__':main()
