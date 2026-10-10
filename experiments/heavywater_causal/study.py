#!/usr/bin/env python3
"""Frozen, calibration-source-only controls for the HeavyWater generation pipeline."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/runs/heavywater_controls'
BASE = ROOT / 'scratch/clean80_english20_test60_plus3wm'
DOMAINS = ('c4', 'squad', 'writing', 'xsum')
MODEL = Path('models/opt-2.7B')
ARMS = [
    dict(name='plain_topk50', top_k=50, repetition=1., ngram=0, mechanism='none', min_new=64),
    dict(name='plain_full', top_k=0, repetition=1., ngram=0, mechanism='none', min_new=64),
    dict(name='plain_repetition', top_k=0, repetition=1.1, ngram=0, mechanism='none', min_new=64),
    dict(name='plain_constraints', top_k=0, repetition=1.1, ngram=4, mechanism='none', min_new=64),
    dict(name='processed_input', top_k=0, repetition=1.1, ngram=4, mechanism='nucleus', min_new=64),
    dict(name='hw_fixed', top_k=0, repetition=1.1, ngram=4, mechanism='fixed', min_new=64),
    dict(name='hw_fresh', top_k=0, repetition=1.1, ngram=4, mechanism='fresh', min_new=64),
    dict(name='hw_fixed_no_constraints', top_k=0, repetition=1., ngram=0, mechanism='fixed', min_new=64),
    dict(name='plain_topk50_stop0', top_k=50, repetition=1., ngram=0, mechanism='none', min_new=0),
]

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def hashtext(s): return hashlib.sha256(s.encode()).hexdigest()
def dump(p, d):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, p)

def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'PROTOCOL.json').exists(): raise RuntimeError('Frozen protocol exists; do not overwrite.')
    regpath = ROOT/'scratch/clean20_complete_regeneration/source_registry.json'
    registry = json.loads(regpath.read_text())
    byid = {r['source_id']: r for r in registry.values()}
    splitpath = ROOT/'configs/original/reference_shaping.json'
    split = json.loads(splitpath.read_text())
    files = {str(regpath): sha(regpath), str(splitpath): sha(splitpath)}
    records = []
    for ds in DOMAINS:
        c = [c for c in split['data_cells'] if c['dataset']==ds and c['generator']=='opt-2.7B']
        assert len(c)==1 and c[0]['split']=='development'
        datapath = Path(c[0]['data_path']); assert sha(datapath)==c[0]['sha256']
        data = json.loads(datapath.read_text())
        mp = ROOT/f'scratch/clean20_all_anomalies_repaired/complete_prompt_free/{ds}/opt-2.7B/manifest.json'
        ids = json.loads(mp.read_text())['source_ids']
        files[str(datapath)] = sha(datapath); files[str(mp)] = sha(mp)
        candidates = []
        for i, sid in enumerate(ids):
            if sid not in byid: continue
            r = byid[sid]
            # Verify identity against the human text, never select by a detector score.
            if ' '.join(r['human_continuation'].split()) != ' '.join(data['original'][i].split()): continue
            candidates.append(dict(dataset=ds, source_id=sid, slot_index=i, prompt=r['prompt'],
                                   human=data['original'][i], saved_unwatermarked=data['unwatermarked'][i],
                                   saved_heavywater=data['HeavyWater'][i], prompt_sha256=hashtext(r['prompt'])))
        candidates.sort(key=lambda r: hashtext('hw-causal-20260917|'+r['source_id']))
        assert len(candidates)>=10, (ds, len(candidates))
        records.extend(candidates[:10])
    sourcefile = OUT/'SOURCES.json'; dump(sourcefile, records)
    protocol = dict(schema='heavywater-causal-controls-v1', created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        source_count=40, domains=list(DOMAINS), calibration_cells_used=4, parent_calibration_cells=20,
        no_new_test_generations=True, detector_parameters_changed=False, selection='first 10 SHA256-ordered identity-verified sources per calibration domain',
        model=str(MODEL), max_new_tokens=200, arms=ARMS, seeds='SHA256(20260917|dataset|source_id); same token-sampling seed across arms, independent state RNG',
        scoring='Original OPT R0^U, higher-is-machine; serialized continuation-only score, raw-token continuation-only score, and exact prompt-conditioned score. Human comparisons use same score view.',
        primary_contrasts=[['plain_full','plain_topk50'],['plain_repetition','plain_full'],['plain_constraints','plain_repetition'],['processed_input','plain_constraints'],['hw_fixed','processed_input'],['hw_fresh','processed_input'],['hw_fixed','hw_fresh'],['hw_fixed','hw_fixed_no_constraints'],['plain_topk50','plain_topk50_stop0']],
        no_outcome_filtering=True, no_retry_selection=True, quality_flags_recorded_only=True,
        scope='Controlled diagnostic on 40 calibration-domain prompts. Does not recreate historical token traces or replace the 60-cell benchmark.',
        source_files=files, sources_sha256=sha(sourcefile), script_sha256=sha(__file__))
    dump(OUT/'PROTOCOL.json',protocol)
    print(json.dumps({k:protocol[k] for k in ['source_count','domains','calibration_cells_used','no_new_test_generations','sources_sha256']},indent=2))

def worker(shard, gpu):
    os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
    os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
    import numpy as np
    import torch
    import transformers
    import ot
    from transformers import LogitsProcessorList, GenerationConfig
    sys.path.insert(0,str(ROOT/'scripts'))
    from run_clean20_watermark_extension_worker import load_model, build_heavywater, normalize_text, sanitize_text, strict_text_quality_reasons
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable in this execution environment.')
    torch.set_num_threads(4)
    protocol=json.loads((OUT/'PROTOCOL.json').read_text())
    assert protocol['script_sha256']==sha(__file__)
    assert protocol['sources_sha256']==sha(OUT/'SOURCES.json')
    records=[r for r in json.loads((OUT/'SOURCES.json').read_text()) if r['dataset']==DOMAINS[shard]]
    work=OUT/f'shard_{shard}';work.mkdir(exist_ok=True)
    model,tok=load_model(MODEL,200)
    engine=build_heavywater(model,tok)
    from watermark.old_watermark import top_p_indices
    device=model.device
    config=GenerationConfig.from_dict(model.generation_config.to_dict())
    dump(work/'RUNTIME.json',dict(python=sys.version,torch=torch.__version__,transformers=transformers.__version__,POT=ot.__version__,gpu=gpu,
         inherited_generation_config=config.to_dict(), model_config_sha256={p.name:sha(p) for p in MODEL.glob('*.json')},
         engine=dict(tilt=engine.tilt, top_p=engine.top_p, k=engine.k, dynamic_seed=engine.dynamic_seed, sinkhorn_reg=engine.sinkhorn_reg, sinkhorn_thresh=engine.sinkhorn_thresh)))

    class ControlledProcessor:
        def __init__(self,mode,seed):
            self.mode=mode; self.rng=torch.Generator(device=device).manual_seed(seed)
            self.calls=0;self.states=[];self.previous=[];self.support=[]
        def __call__(self,ids,scores):
            assert ids.shape[0]==1, 'Upstream processor reuses the last state for batches; require batch size one.'
            self.calls+=1
            prev=int(ids[0,-1]);self.previous.append(prev)
            p=scores[0].float().softmax(-1)
            keep=top_p_indices(p,engine.top_p);self.support.append(len(keep))
            if self.mode=='fixed':
                state_rng=torch.Generator(device=device).manual_seed(engine.large_prime*prev)
                self.states.append(int(torch.randint(1,engine.k+1,(1,),generator=state_rng,device=device))-1)
                return engine(ids,scores)
            if self.mode=='fresh':
                state=int(torch.randint(1,engine.k+1,(1,),generator=self.rng,device=device))
                self.states.append(state-1)
                return engine.tilt_q_HT(p,state).log().unsqueeze(0)
            assert self.mode=='nucleus'
            # Match upstream singleton passthrough exactly.
            if len(keep)==1:return scores
            q=torch.zeros_like(p);q[keep]=p[keep]/p[keep].sum()
            return q.log().unsqueeze(0)

    @torch.inference_mode()
    def score(ids,start=1):
        if ids.numel()<2 or start>=ids.numel():return None
        logits=model(input_ids=ids.unsqueeze(0),use_cache=False).logits[0,start-1:-1].float()
        ell=logits.log_softmax(-1);p=ell.exp();target=ids[start:]
        mu=(p*ell).sum(-1);nu=((p*ell.square()).sum(-1)-mu.square()).clamp_min(0)
        obs=ell.gather(-1,target[:,None]).squeeze(-1);res=obs-mu
        # Float64 aggregation; reference moments use the model's Float32 logits.
        rr=res.double().cpu().numpy();vv=nu.double().cpu().numpy()
        def part(n):
            nn=min(n,len(rr));return dict(n=nn,R=float(rr[:nn].sum()/max(float(np.sqrt(vv[:nn].sum())),nn*1e-9)),
                 residual_mean=float(rr[:nn].mean()), variance_mean=float(vv[:nn].mean()))
        return dict(full=part(len(rr)),first64=part(64),first128=part(128),residuals=rr.tolist(),variances=vv.tolist())

    def bareids(g):
        prefix=[] if tok.bos_token_id is None else [tok.bos_token_id]
        return torch.tensor(prefix+g.tolist(),device=device,dtype=torch.long)

    for ri,r in enumerate(records):
        prefix=tok(r['prompt'],return_tensors='pt').input_ids[0].to(device)
        assert len(prefix)+200<=model.config.max_position_embeddings
        sourcekey=hashtext(r['source_id'])[:16]
        token_seed=int(hashtext('20260917|'+r['dataset']+'|'+r['source_id'])[:8],16)
        reference_path=work/f'{sourcekey}_reference.json'
        if not reference_path.exists():
            refs={}
            for label,tx in [('human',r['human']),('saved_unwatermarked',r['saved_unwatermarked']),('saved_heavywater',r['saved_heavywater'])]:
                ids=tok(tx,return_tensors='pt').input_ids[0].to(device)[:2048]
                content=tok(tx,add_special_tokens=False,return_tensors='pt').input_ids[0].to(device)
                both=torch.cat([prefix,content])[:2048]
                refs[label]=dict(serialized=score(ids),prompt_conditioned=score(both,len(prefix)))
            dump(reference_path,dict(dataset=r['dataset'],source_id=r['source_id'],scores=refs))
        for arm in ARMS:
            dest=work/f'{sourcekey}_{arm["name"]}.json'
            if dest.exists():continue
            torch.manual_seed(token_seed);torch.cuda.manual_seed_all(token_seed)
            engine.seed_increment=0
            proc=None if arm['mechanism']=='none' else ControlledProcessor(arm['mechanism'], token_seed^0x63a79b2d)
            before=time.time()
            params=dict(max_new_tokens=200,min_new_tokens=arm['min_new'],do_sample=True,num_beams=1,
                temperature=1.,top_k=arm['top_k'],top_p=1.,repetition_penalty=arm['repetition'],no_repeat_ngram_size=arm['ngram'],
                pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,use_cache=True,
                logits_processor=LogitsProcessorList([] if proc is None else [proc]))
            with torch.inference_mode():out=model.generate(input_ids=prefix[None],attention_mask=torch.ones_like(prefix[None]),**params)[0]
            generated=out[len(prefix):]
            raw=tok.decode(generated,skip_special_tokens=True)
            clean,removed=sanitize_text(raw);serialized=normalize_text(clean)
            # Save actual original generated IDs and compare all views without changing tokens.
            visible=generated
            if len(visible) and int(visible[-1])==tok.eos_token_id: visible=visible[:-1]
            ids=tok(serialized,return_tensors='pt').input_ids[0].to(device)[:2048]
            result=dict(dataset=r['dataset'],source_id=r['source_id'],arm=arm,seed=token_seed,
                seconds=time.time()-before,prompt_ids=prefix.tolist(),generated_ids=generated.tolist(),raw=raw,serialized=serialized,
                removed_chars=removed,quality_flags=strict_text_quality_reasons(serialized),no_resampling=True,
                scores=dict(serialized=score(ids),raw_ids=score(bareids(visible)),prompt_conditioned=score(torch.cat([prefix,visible]),len(prefix))),
                state_trace=None if proc is None else dict(states=proc.states,previous=proc.previous,support=proc.support,calls=proc.calls))
            dump(dest,result)
            print(f'{r["dataset"]} {ri+1}/10 {arm["name"]}: tokens={len(generated)}, R={result["scores"]["serialized"]["full"]["R"] if result["scores"]["serialized"] else None}, {result["seconds"]:.1f}s',flush=True)
        dump(work/'PROGRESS.json',dict(completed_sources=ri+1,expected_sources=10,done=ri==9))
    dump(work/'DONE.json',dict(status='PASS',sources=len(records),arms=len(ARMS)))

def launch():
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_PROGRESS_BARS='1')
    ps=[];handles=[]
    for shard in range(4):
        log=(OUT/f'worker_{shard}.log').open('a');handles.append(log)
        cmd=[sys.executable,str(Path(__file__).resolve()),'worker','--shard',str(shard),'--gpu',str(shard)]
        ps.append(subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT))
    dump(OUT/'LAUNCH.json',dict(pids=[p.pid for p in ps],gpus=list(range(4)),script_sha256=sha(__file__)))
    while any(p.poll() is None for p in ps):
        time.sleep(20)
        status=[]
        for i,p in enumerate(ps):
            d=OUT/f'shard_{i}/PROGRESS.json'
            status.append(dict(shard=i,exit=p.poll(),progress=json.loads(d.read_text()) if d.exists() else None))
        print(json.dumps(status),flush=True)
    codes=[p.returncode for p in ps]
    for h in handles:h.close()
    dump(OUT/'LAUNCH_EXIT.json',dict(exit_codes=codes))
    if any(codes):raise RuntimeError(codes)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','launch','worker'])
    p.add_argument('--shard',type=int);p.add_argument('--gpu',type=int)
    a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='launch':launch()
    else:worker(a.shard,a.gpu)
