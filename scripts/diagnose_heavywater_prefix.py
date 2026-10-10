#!/usr/bin/env python3
"""Bounded actual no-tilt processor replay on frozen reconstructed OPT prefixes."""
import argparse
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["HF_HUB_OFFLINE"]="1"
os.environ["TRANSFORMERS_OFFLINE"]="1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"]="1"
from pathlib import Path
import hashlib,json,warnings,time
import numpy as np
import pandas as pd
import torch
import ot
from transformers import RepetitionPenaltyLogitsProcessor,NoRepeatNGramLogitsProcessor,TopKLogitsWarper,TopPLogitsWarper,GenerationConfig
from run_clean20_watermark_extension_worker import load_model,build_heavywater

ROOT=Path(__file__).resolve().parents[1]
BASE=Path("scratch/clean80_english20_test60_plus3wm")
OUT=Path("scratch/heavywater_prefix")
MODEL=Path("models/opt-2.7B")
PROTOCOL=ROOT/"results/runs/heavywater_prefix/FROZEN_PROTOCOL.md"
DOMAINS=("cmv","cnn","hswag","sci")
POSITIONS=(64,96,128,160)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()

def dump(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")

def expect(p,ell):return float(torch.dot(p.double(),ell.double()).item())

def main():
    global BASE, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE,
                        help="Prepared source workspace containing requests and dataset/")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    for path in (args.base, args.output):
        if path.is_absolute() or ".." in path.parts:
            parser.error("Paths must be relative to the repository root")
    if not args.output.parts or args.output.parts[0] != "scratch":
        parser.error("--output must be a new directory below scratch/")
    BASE, OUT = args.base, args.output
    if not torch.cuda.is_available():raise RuntimeError("CUDA inaccessible; no CPU or alternate-device fallback.")
    if OUT.exists():raise RuntimeError("Output exists; refuse to overwrite frozen run.")
    OUT.mkdir(parents=True)
    torch.set_num_threads(4)
    source=BASE/"extension/generation_requests.json"
    requests=json.loads(source.read_text())
    selected=[]
    for ds in DOMAINS:
        req=[a for a in requests if a["dataset"]==ds and a["generator"]=="opt-2.7B" and a["method"]=="HeavyWater" and a["slot_index"]==0]
        assert len(req)==1
        req=req[0]
        assert hashlib.sha256(req["prompt"].encode()).hexdigest()==req["prompt_sha256"]
        datafile=BASE/f"dataset/complete_prompt_free/{ds}/opt-2.7B/data.json"
        data=json.loads(datafile.read_text())
        for mechanism in ("unwatermarked","HeavyWater"):
            selected.append({"dataset":ds,"mechanism":mechanism,"source_id":req["source_id"],
                             "prompt":req["prompt"],"continuation":data[mechanism][0],
                             "prompt_sha256":req["prompt_sha256"],"datafile":str(datafile),"data_sha256":sha(datafile)})
    freeze={"protocol_sha256":sha(PROTOCOL),"script_sha256":sha(__file__),"requests_sha256":sha(source),
            "source_metadata":[{k:v for k,v in a.items() if k not in ("prompt","continuation")} for a in selected],
            "positions":POSITIONS,"cuda_visible_devices":os.environ["CUDA_VISIBLE_DEVICES"],
            "torch":torch.__version__,"POT":ot.__version__,
            "reconstructed_prefixes_not_original_token_trace":True,"new_generations":False,
            "code_sha256":{str(p.relative_to(ROOT)):sha(p) for p in (
                ROOT/"scripts/run_clean20_watermark_extension_worker.py",
                ROOT/"scripts/generation_utils.py", ROOT/"scripts/text_utils.py",
                ROOT/"scripts/io_utils.py", ROOT/"scripts/release_utils.py",
                ROOT/"third_party/external_watermarks/HeavyWater_SimplexWater/watermark/heavy_tail_randscore.py")},
            "model_config_sha256":{p.name:sha(p) for p in MODEL.glob("*.json")},
            "model_weight_sha256":{p.name:sha(p) for p in sorted(MODEL.iterdir())
                                   if p.suffix in (".bin",".safetensors")}}
    dump("FROZEN.json",freeze)
    print("Loading local OPT-2.7B; at most 32 fixed prefixes",flush=True)
    model,tok=load_model(MODEL,200)
    engine=build_heavywater(model,tok)
    assert engine.tilt is False and engine.tilting_delta==0 and engine.k==1024
    from watermark.old_watermark import top_p_indices
    rep=RepetitionPenaltyLogitsProcessor(1.1)
    ngram=NoRepeatNGramLogitsProcessor(4)
    cfg=model.generation_config
    resolved=GenerationConfig.from_dict(cfg.to_dict())
    resolved.update(do_sample=True,temperature=1.0,max_new_tokens=200)
    top_k=resolved.top_k
    top_p=resolved.top_p
    assert top_k==50 and top_p==1.0
    dump("BASELINE_RECONSTRUCTION.json",{"loaded_generation_config":cfg.to_dict(),
        "effective_top_k_reconstruction":top_k,"effective_top_p_reconstruction":top_p,
        "scope":"Current inherited values only; not certified historical runtime."})
    original_sinkhorn=ot.sinkhorn
    capture={}
    def instrumented(*args,**kwargs):
        assert not args
        call=dict(kwargs);call["log"]=True
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            plan,log=original_sinkhorn(**call)
        capture.clear()
        capture.update(plan=plan,a=kwargs["a"],b=kwargs["b"],cost=kwargs["M"],log=log,
                       warnings=[str(w.message) for w in caught])
        return plan
    ot.sinkhorn=instrumented
    rows,skips,snapshots,state_records=[],[],[],[]
    started=time.time()
    for item in selected:
        pids=tok(item["prompt"],return_tensors="pt").input_ids[0]
        cfull=tok(item["continuation"],return_tensors="pt").input_ids[0]
        # Obtain the exact continuation-only tokenization used as the detector view.
        prefix_special=[]
        if len(cfull) and tok.bos_token_id is not None and int(cfull[0])==tok.bos_token_id:
            prefix_special=[int(cfull[0])];cids=cfull[1:]
        else:cids=cfull
        valid=[]
        for n in POSITIONS:
            if len(cids)<=n:
                skips.append({"dataset":item["dataset"],"mechanism":item["mechanism"],"position":n,"reason":"continuation_too_short"});continue
            if len(pids)+n>model.config.max_position_embeddings:
                skips.append({"dataset":item["dataset"],"mechanism":item["mechanism"],"position":n,"reason":"context_capacity"});continue
            valid.append(n)
        if not valid:continue
        longest=max(valid)
        full=torch.cat([pids,cids[:longest]]).unsqueeze(0).to(model.device)
        bare=torch.tensor(prefix_special+cids[:longest].tolist(),device=model.device).unsqueeze(0)
        with torch.inference_mode():
            gen_logits=model(input_ids=full,use_cache=False).logits[0]
            det_logits=model(input_ids=bare,use_cache=False).logits[0]
        for n in valid:
            meta={k:item[k] for k in ("dataset","mechanism","source_id","prompt_sha256")}
            meta.update(position=n,prompt_tokens=len(pids),continuation_tokens=len(cids))
            ids=full[:,:len(pids)+n]
            raw=gen_logits[len(pids)+n-1].float().clone()
            det=det_logits[len(prefix_special)+n-1].float().clone()
            ell=det.log_softmax(-1);pdet=det.softmax(-1);praw=raw.softmax(-1)
            baseline=raw.clone().unsqueeze(0)
            if top_k and top_k>0:baseline=TopKLogitsWarper(top_k)(ids,baseline)
            if top_p and top_p<1:baseline=TopPLogitsWarper(top_p)(ids,baseline)
            pbase=baseline[0].softmax(-1)
            rep_logits=rep(ids,raw.clone().unsqueeze(0))
            prep=rep_logits[0].softmax(-1)
            processed=ngram(ids,rep_logits.clone())
            pprocessed=processed[0].softmax(-1)
            keep=top_p_indices(pprocessed,engine.top_p)
            a=pprocessed[keep];a=a/a.sum()
            # Reproduce the actual CUDA state RNG without consuming global sampling RNG.
            state_rng=torch.Generator(device=ids.device)
            state_rng.manual_seed(engine.large_prime*int(ids[0,-1]))
            state=int(torch.randint(1,engine.k+1,(1,),generator=state_rng,device=ids.device).item())-1
            capture.clear()
            outlog=engine(ids,processed.clone())[0]
            pout=outlog.exp()
            eraw=expect(praw,ell);edet=expect(pdet,ell);ebase=expect(pbase,ell)
            erep=expect(prep,ell);eproc=expect(pprocessed,ell);esel=expect(pout,ell)
            extra={}
            if len(keep)==1:
                enucleus=eproc;eavg=eproc;state_delta=0.
                shifts=np.zeros(engine.k)
                extra.update(singleton_passthrough=True,numeric_row_l1=0.,numeric_col_l1=0.,uniform_l1=0.,
                             marginal_bound_slack=0.,selected_parity_l1=float((pout-pprocessed).abs().sum()),
                             selected_float64_l1=0.,selected_float64_score_difference=0.,sinkhorn_iterations=0,
                             sinkhorn_warning="",float64_warning="")
            else:
                plan=capture["plan"];target=capture["a"];b=capture["b"]
                colmass=plan.sum(0)
                assert bool((colmass>0).all())
                conditional=plan/colmass.unsqueeze(0)
                avg=conditional.mean(1)
                feature=ell[keep].double()
                allmeans=conditional.double().T@feature
                enucleus=expect(target,ell[keep]);eavg=float(allmeans.mean())
                state_delta=esel-eavg
                shifts=(allmeans-enucleus).detach().cpu().numpy()
                direct=conditional[:,state]
                rowerr=float((plan.sum(1)-target).abs().sum())
                colerr=float((colmass-b).abs().sum())
                avgerr=float((avg-target).abs().sum())
                # Tighter numerical control for the identical captured OT problem.
                a64=target.double();a64=a64/a64.sum()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    p64,log64=original_sinkhorn(a64,b.double(),capture["cost"].double(),reg=.05,
                                                numItermax=5000,stopThr=1e-10,log=True)
                q64=p64[:,state]/p64[:,state].sum()
                extra.update(singleton_passthrough=False,numeric_row_l1=rowerr,numeric_col_l1=colerr,
                    uniform_l1=avgerr,marginal_bound_slack=rowerr+colerr-avgerr,
                    selected_parity_l1=float((pout[keep]-direct).abs().sum()),
                    selected_float64_l1=float((direct.double()-q64).abs().sum()),
                    selected_float64_score_difference=expect(direct-q64,ell[keep]),
                    sinkhorn_iterations=int(capture["log"].get("niter",-1)),
                    sinkhorn_warning=" | ".join(capture["warnings"]),
                    float64_warning=" | ".join(str(w.message) for w in caught),
                    float64_iterations=int(log64.get("niter",-1)),
                    float64_row_l1=float((p64.sum(1)-a64).abs().sum()),
                    float64_col_l1=float((p64.sum(0)-b.double()).abs().sum()))
            terms={"prompt_context":eraw-edet,"repetition":erep-eraw,"ngram":eproc-erep,
                   "nucleus":enucleus-eproc,"ot_uniform_numerical":eavg-enucleus,
                   "selected_state":state_delta}
            total=esel-edet
            assert abs(sum(terms.values())-total)<2e-10
            reference_var=expect(pdet,(ell-edet)**2)
            row={**meta,**extra,"selected_state_id":state,"support":len(keep),
                 "removed_mass":float(1-pprocessed[keep].sum()),
                 **{"delta_"+k:v for k,v in terms.items()},
                 "delta_total_to_detector_reference":total,"delta_to_current_baseline":esel-ebase,
                 "delta_current_baseline_to_raw":ebase-eraw,
                 "observed_next_residual":float(ell[int(cids[n])])-edet,
                 "reference_sd":float(np.sqrt(reference_var)),
                 "all_state_delta_min":float(shifts.min()),"all_state_delta_max":float(shifts.max()),
                 "all_state_delta_q05":float(np.quantile(shifts,.05)),
                 "all_state_delta_median":float(np.median(shifts)),
                 "all_state_delta_q95":float(np.quantile(shifts,.95)),
                 "all_state_negative_fraction":float((shifts<0).mean()),
                 "telescoping_error":sum(terms.values())-total}
            rows.append(row)
            state_records.append({"metadata":meta,"all_state_delta_from_nucleus":shifts})
            snapshots.append({"metadata":meta,"full_prefix_ids":ids[0].cpu(),
                              "raw_generator_logits":raw.cpu(),"detector_logits":det.cpu(),
                              "observed_next_id":int(cids[n]),"selected_state_id":state})
            pd.DataFrame(rows).to_csv(OUT/"prefix_metrics.csv",index=False)
            print(f'{item["dataset"]} {item["mechanism"]} n={n}: total={total:.5f} rep={terms["repetition"]:.5f} state={state_delta:.5f} numerical={terms["ot_uniform_numerical"]:.2g}',flush=True)
        del gen_logits,det_logits
    ot.sinkhorn=original_sinkhorn
    torch.save(snapshots,OUT/"logits_snapshots.pt")
    torch.save(state_records,OUT/"all_state_shifts.pt")
    dump("SKIPS.json",skips)
    df=pd.DataFrame(rows)
    assert len(df)>0
    cols=[k for k in df if k.startswith("delta_")]+["numeric_row_l1","numeric_col_l1","uniform_l1",
         "selected_float64_l1","selected_float64_score_difference","selected_parity_l1","observed_next_residual"]
    df.groupby("mechanism")[cols].agg(["mean","median","min","max"]).to_csv(OUT/"summary_by_mechanism.csv")
    df.groupby(["dataset","mechanism"])[cols].mean().to_csv(OUT/"summary_by_dataset.csv")
    dump("SUMMARY.json",{"prefixes":len(df),"skipped":len(skips),"runtime_seconds":time.time()-started,
         "mean_components":{k:float(df[k].mean()) for k in cols},
         "max_telescoping_error":float(df.telescoping_error.abs().max()),
         "max_implementation_parity_l1":float(df.selected_parity_l1.max()),
         "max_selected_float64_l1":float(df.selected_float64_l1.max()),
         "warnings":df.loc[df.sinkhorn_warning.ne("")|df.float64_warning.ne(""),
                           ["dataset","mechanism","position","sinkhorn_warning","float64_warning"]].to_dict("records")})
    print("Completed fixed-prefix diagnostic",flush=True)

if __name__=="__main__":main()
