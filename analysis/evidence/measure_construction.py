#!/usr/bin/env python3
"""Fresh OPT measurements of the current upper response at local T and c settings.

Prepare identifiers before inference; each worker handles an immutable shard.
Raw text stays in the existing dataset and is not copied into the paper package.
"""
import argparse, hashlib, json, math, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
SETTINGS = [(1.1,.8),(1.2,.7),(1.2,.8),(1.2,.9),(1.3,.8)]

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def prepare(project, output):
    import pandas as pd
    base = project/'results/runs/calibrated_scores'
    cal = pd.read_pickle(base/'calibration_OPT.pkl')
    query = pd.read_pickle(base/'test/OPT/scoring_frame.pkl')
    humans = lambda q: q[q.label == 0].sort_values(['dataset','source_id','generator']).drop_duplicates(['dataset','source_id'])
    cal = humans(cal); human = humans(query)
    selected = set()
    for ds, block in human.groupby('dataset'):
        sources = sorted(block.source_id, key=lambda s: hashlib.sha256(('kappa-v8-20260913::'+s).encode()).hexdigest())
        selected.update(sources)
    machine = query[(query.label == 1) & query.source_id.isin(selected)]
    assert (len(cal),len(human),len(machine)) == (800,2360,129800)
    frame = pd.concat([cal,human,machine],ignore_index=True).sort_values('row_id').reset_index(drop=True)
    cfg = json.loads((project/'configs/original/reference_shaping.json').read_text())
    cells = {(c['dataset'],c['generator']): c for c in cfg['data_cells']}
    rows=[]
    for i,r in enumerate(frame.to_dict('records')):
        c=cells[(r['dataset'],r['generator'])]
        rows.append({**{k:r[k] for k in ['row_id','source_id','dataset','generator','watermark_method','label','split','text_sha256','token_length']},
                     'input_path':c['data_path'],'input_sha256':c['sha256'],
                     'source_index':int(r['source_id'].split(':')[-1]),'shard':i%7,
                     'cached_upper':float(r['entropy_N1']/max(math.sqrt(max(r['entropy_V1'],0)),1e-9)),
                     'cached_lower':float(r['uniform_N0']/max(math.sqrt(max(r['uniform_V0'],0)),1e-9))})
    output.mkdir(parents=True,exist_ok=True)
    with (output/'cohort.jsonl').open('x') as f:
        for row in rows:f.write(json.dumps(row)+'\n')
    manifest={'rows':len(rows),'calibration_humans':800,'evaluation_humans':2360,'evaluation_machines':129800,'selected_machine_sources':sorted(selected),'settings':SETTINGS,'shards':7,'cohort_sha256':sha(output/'cohort.jsonl'),'protocol_sha256':sha(HERE/'evidence/LOCAL_CONSTRUCTION_PROTOCOL.json'),'script_sha256':sha(__file__),'source_config_sha256':sha(project/'configs/original/reference_shaping.json')}
    (output/'cohort_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))

def features(logits, labels):
    import torch
    target, entropy, fullmeans, fullvars = [], [], [], []
    means={s:[] for s in SETTINGS};variances={s:[] for s in SETTINGS}
    D=logits.shape[-1]
    for start in range(0,len(labels),128):
        z=logits[start:start+128].float();y=labels[start:start+128]
        lp=torch.log_softmax(z,dim=-1);p=lp.exp()
        target.append(lp.gather(-1,y[:,None]).squeeze(-1).cpu())
        entropy.append((-(p*lp).sum(-1)/math.log(D)).cpu())
        p0=torch.softmax(z,dim=-1);mu0=(p0*lp).sum(-1)
        fullmeans.append(mu0.cpu());fullvars.append(((p0*lp.square()).sum(-1)-mu0.square()).clamp_min(0).cpu())
        for T in sorted({s[0] for s in SETTINGS}):
            qt=torch.softmax(z/T,dim=-1);sq,indices=torch.sort(qt,dim=-1,descending=True)
            slp=lp.gather(-1,indices);cumulative=sq.cumsum(-1)
            for c in sorted({s[1] for s in SETTINGS if s[0]==T}):
                keep=torch.ones_like(sq,dtype=torch.bool);keep[:,1:]=~(cumulative>c)[:,:-1]
                vals=sq*keep;dist=vals/vals.sum(-1,keepdim=True)
                mu=(dist*slp).sum(-1)
                means[T,c].append(mu.cpu());variances[T,c].append(((dist*slp.square()).sum(-1)-mu.square()).clamp_min(0).cpu())
    target=torch.cat(target);h=torch.cat(entropy)
    cap=max(float(h.mean())*1.5,1e-6);a=h.clamp(max=cap);w=a/a.sum().clamp_min(1e-8)
    result={}
    for T,c in SETTINGS:
        mu=torch.cat(means[T,c]);nu=torch.cat(variances[T,c]);N=float((w*target).sum()-(w*mu).sum());V=float((w.square()*nu).sum())
        result[str(T)+'_'+str(c)]={'upper':N/max(math.sqrt(max(V,0)),1e-9),'neff':float(1/w.square().sum())}
    w=torch.ones_like(h)/len(h);mu0=torch.cat(fullmeans);nu0=torch.cat(fullvars)
    lower=float((w*target).sum()-(w*mu0).sum())/max(math.sqrt(float((w.square()*nu0).sum())),1e-9)
    return {'settings':result,'recomputed_lower':lower,'entropy_mean':float(h.mean()),'positions':len(h)}

def self_check(project):
    import torch
    sys.path.insert(0,str(project/'scripts'))
    from score_reference_moments import features as reference
    torch.manual_seed(20260913)
    for n,D in [(1,7),(17,31),(257,23)]:
        z=torch.randn(n,D)*3;y=torch.randint(D,(n,))
        a=features(z,y);b=reference(z,z,y)
        s=b['stats']['entropy'];expected=s['N1']/max(math.sqrt(s['V1']),1e-9)
        assert abs(a['settings']['1.2_0.8']['upper']-expected)<1e-9
        assert abs(a['settings']['1.2_0.8']['neff']-s['neff'])<1e-9
        s=b['stats']['uniform'];expected=s['N0']/max(math.sqrt(s['V0']),1e-9)
        assert abs(a['recomputed_lower']-expected)<1e-9
        assert all(0<r['neff']<=n+1e-4 for r in a['settings'].values())
    print('PASS: matched current moments, cap, floors and default on three synthetic sequences',flush=True)

def run(project,output,shard):
    import torch
    sys.path.insert(0,str(project/'scripts'))
    from model import load_model,load_tokenizer
    torch.set_num_threads(2)
    manifest=json.loads((output/'cohort_manifest.json').read_text())
    assert sha(output/'cohort.jsonl')==manifest['cohort_sha256']
    assert sha(__file__)==manifest['script_sha256']
    assert sha(HERE/'evidence/LOCAL_CONSTRUCTION_PROTOCOL.json')==manifest['protocol_sha256']
    self_check(project)
    modelpath='models/opt-2.7B'
    tok=load_tokenizer(modelpath,str(Path(modelpath).parent))
    model=load_model(modelpath,'cuda:0',str(Path(modelpath).parent)).eval();model.config.use_cache=False
    rows=[json.loads(line) for line in (output/'cohort.jsonl').read_text().splitlines()]
    rows=[r for r in rows if r['shard']==shard]
    records=[]
    for path in sorted({r['input_path'] for r in rows}):
        group=[r for r in rows if r['input_path']==path]
        assert sha(path)==group[0]['input_sha256']
        raw=json.loads(Path(path).read_text())
        for r in group:
            text=raw['original' if r['label']==0 else r['watermark_method']][r['source_index']]
            assert hashlib.sha256(text.encode()).hexdigest()==r['text_sha256']
            ids=tok(text,add_special_tokens=True,return_token_type_ids=False)['input_ids'][:2048]
            assert len(ids)-1==r['token_length']
            records.append((ids,r))
    records.sort(key=lambda r:(len(r[0]),r[1]['row_id']))
    offset=0;started=time.monotonic();maxerr=0
    partial=output/('shard_%d.partial.jsonl'%shard);final=output/('shard_%d.jsonl'%shard)
    if final.exists():raise FileExistsError(final)
    with partial.open('x') as f:
        while offset<len(records):
            stop=min(offset+8,len(records))
            while stop>offset+1 and max(len(x[0]) for x in records[offset:stop])*(stop-offset)>2048:stop-=1
            batch=records[offset:stop]
            encoded=tok.pad([{'input_ids':ids,'attention_mask':[1]*len(ids)} for ids,_ in batch],return_tensors='pt').to('cuda:0')
            with torch.inference_mode():logits=model(**encoded,use_cache=False).logits
            for pos,(ids,r) in enumerate(batch):
                result=features(logits[pos,:len(ids)-1],encoded['input_ids'][pos,1:len(ids)])
                # Raw source locations are kept only in the local cohort manifest.
                row={k:v for k,v in r.items() if k not in ['input_path','input_sha256']}
                maxerr=max(maxerr,abs(result['settings']['1.2_0.8']['upper']-r['cached_upper']))
                f.write(json.dumps({**row,**result},separators=(',',':'),allow_nan=False)+'\n')
            del logits
            offset=stop
            if offset%200<8 or offset==len(records):
                f.flush();print(json.dumps({'shard':shard,'rows':offset,'total':len(records),'seconds':round(time.monotonic()-started,1),'max_default_forward_difference':maxerr}),flush=True)
    partial.rename(final)
    audit={'status':'PASS','shard':shard,'rows':len(records),'seconds':time.monotonic()-started,'max_default_forward_difference':maxerr,'sha256':sha(final),'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0),'gpu_visibility':os.environ.get('CUDA_VISIBLE_DEVICES'),'cohort_sha256':manifest['cohort_sha256'],'script_sha256':sha(__file__)}
    (output/('shard_%d.audit.json'%shard)).write_text(json.dumps(audit,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','check']);p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--shard',type=int,default=0);args=p.parse_args()
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    if args.mode=='prepare':prepare(args.project,args.output)
    elif args.mode=='check':self_check(args.project)
    else:run(args.project,args.output,args.shard)
