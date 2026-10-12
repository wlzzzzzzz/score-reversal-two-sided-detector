"""Frozen paired-model scoring for the paper baseline comparisons.

One model per GPU for Falcon; all statistics share the same forward passes.
Cell files are immutable and resumable only after input/config hash validation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from model import load_model, load_tokenizer
from fastdetect_official_module_core import _document_score, _weights, OfficialModuleConfig
from fastdetect_official_module_chunked import compute_official_module_features_token_chunked
from fast_detect_gpt import get_sampling_discrepancy_analytic


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def tag(t, p, weight):
    return "T%s_P%s_W%s" % (str(t).replace(".", "p"), str(p).replace(".", "p"), weight)


def features(ref, score, labels, temperatures, masses, sensitivity=False, chunk=128):
    n = labels.numel()
    common = min(ref.shape[-1], score.shape[-1])
    assert int(labels.max()) < common
    target, entropy = [], []
    means = {(t, p): [] for t in temperatures for p in masses}
    variances = {key: [] for key in means}
    degenerate, outside = {key: 0 for key in means}, {key: 0 for key in means}
    retained, minima = {key: [] for key in means}, {key: common for key in means}
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        r = ref[start:stop, :common].float()
        s = score[start:stop, :common].to(ref.device).float()
        lab = labels[start:stop].to(ref.device)
        lp = torch.log_softmax(s, -1)
        lq = torch.log_softmax(r, -1)
        q = lq.exp()
        target.append(lp.gather(-1, lab[:, None]).squeeze(-1).cpu())
        entropy.append((-(q*lq).sum(-1) / math.log(common)).cpu())
        for t in temperatures:
            qt = torch.softmax(r / t, -1)
            if any(p < 1.0 for p in masses):
                sorted_q, indices = torch.sort(qt, dim=-1, descending=True)
                sorted_lp = lp.gather(-1, indices)
                cumulative = sorted_q.cumsum(-1)
            for p in masses:
                key = (t, p)
                if p == 1.0:
                    dist, logprob = qt, lp
                    mass = torch.ones(len(lab), device=ref.device)
                else:
                    removal = cumulative > p
                    mask = torch.ones_like(removal)
                    mask[:, 1:] = ~removal[:, :-1]
                    values = sorted_q * mask
                    mass = values.sum(-1)
                    dist = values / mass[:, None]
                    logprob = sorted_lp
                    positions = (indices == lab[:, None]).nonzero(as_tuple=False)[:, 1]
                    outside[key] += int((~mask[torch.arange(len(lab), device=ref.device), positions]).sum())
                    minima[key] = min(minima[key], int(mask.sum(-1).min()))
                mu = (dist * logprob).sum(-1)
                var = ((dist * logprob.square()).sum(-1) - mu.square()).clamp_min(0)
                means[key].append(mu.cpu()); variances[key].append(var.cpu())
                degenerate[key] += int((var <= 1e-9).sum())
                retained[key].append(mass.cpu())
    target, entropy = torch.cat(target), torch.cat(entropy)
    weight_map = {"uniform": _weights(entropy, "uniform", 1.5),
                  "entropy": _weights(entropy, "linear_entropy", 1.5)}
    if sensitivity:
        for cap in [1.0, 2.0]:
            weight_map["cap" + str(cap).replace(".", "p")] = _weights(entropy, "linear_entropy", cap)
        weight_map["uncapped"] = entropy / entropy.sum().clamp_min(1e-8)
    result, diag = {}, {}
    for (t, p), values in means.items():
        mu, var = torch.cat(values), torch.cat(variances[(t,p)])
        for wname, w in weight_map.items():
            result[tag(t,p,wname)] = _document_score(target, mu, var, w, 1e-9)
        if (t,p) == (1.0,1.0):
            variance = var.sum()
            if float(variance) <= 0:
                raise RuntimeError("non-positive native Fast-DetectGPT variance")
            result["fastdetect_native"] = float(((target.sum()-mu.sum()) / variance.sqrt()).item())
        if (t,p) == (1.2,0.8):
            w = weight_map["entropy"]
            denominator = float((w.square()*var).sum().sqrt())
            diag = {"conditional_std": denominator,
                    "epsilon_active": denominator < 1e-9,
                    "nucleus_degenerate_fraction": degenerate[(t,p)] / n,
                    "observed_outside_nucleus_fraction": outside[(t,p)] / n,
                    "retained_mass_mean": float(torch.cat(retained[(t,p)]).mean()),
                    "min_support_tokens": minima[(t,p)],
                    "effective_weight_tokens": float(1.0 / w.square().sum()),
                    "raw_residual": float((w*target).sum() - (w*mu).sum())}
    if not all(math.isfinite(value) for value in result.values()):
        raise RuntimeError("non-finite detection statistic")
    return {**result, **diag}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--worker", type=int, required=True)
    ap.add_argument("--smoke-rows", type=int, default=0)
    args = ap.parse_args()
    config = json.loads(args.config.read_text())
    config_hash = sha(args.config)
    worker = config["workers"][args.worker]
    out = Path(config["output_root"])
    out.mkdir(parents=True, exist_ok=True)
    tokenizers = [load_tokenizer(p, str(Path(p).parent)) for p in [worker["q"],worker["p"]]]
    assert tokenizers[0].get_vocab() == tokenizers[1].get_vocab()
    same = worker["q"] == worker["p"]
    qdevice, pdevice = "cuda:0", "cuda:0" if same else "cuda:1"
    qm = load_model(worker["q"], qdevice, str(Path(worker["q"]).parent)).eval()
    pm = qm if same else load_model(worker["p"], pdevice, str(Path(worker["p"]).parent)).eval()
    for model in [qm,pm]: model.config.use_cache = False
    groups = out / (worker["name"] + ("_smoke" if args.smoke_rows else ""))
    groups.mkdir(exist_ok=True)
    max_batch, max_budget = config["batch_size"], config["max_batch_tokens"]
    report = {"worker":args.worker,"status":"RUNNING","rows":0,"cells":0,
              "config_sha256":config_hash,"max_official_parity_error":0.0,
              "max_s3_parity_error":0.0,"forward_seconds":0.0,"module_seconds":0.0}
    start_time = time.perf_counter()
    for cell in worker["cells"]:
        source = Path(cell["data_path"])
        assert sha(source) == cell["sha256"]
        output = groups / (cell["dataset"] + "_" + cell["generator"] + ".jsonl")
        manifest = output.with_suffix(".manifest.json")
        if output.exists():
            old = json.loads(manifest.read_text())
            assert old["config_sha256"] == config_hash and old["sha256"] == sha(output)
            report["rows"] += old["rows"]; report["cells"] += 1
            continue
        data = json.loads(source.read_text())
        count = int(cell["source_count"])
        records = []
        for method in ["human"] + config["methods"]:
            values = data["original" if method == "human" else method]
            assert len(values) == count
            for index, text in enumerate(values):
                ids = tokenizers[0](text, add_special_tokens=True, return_token_type_ids=False)["input_ids"]
                other = tokenizers[1](text, add_special_tokens=True, return_token_type_ids=False)["input_ids"]
                assert ids == other
                length = len(ids)
                ids = ids[:config["max_tokens"]]
                assert len(ids) >= 2
                row = {"row_id":f"{cell['dataset']}:{cell['generator']}:{index}:{method}",
                       "source_id":f"{cell['dataset']}:{index}","source_index":index,
                       "dataset":cell["dataset"],"generator":cell["generator"],
                       "watermark_method":method,"label":int(method != "human"),
                       "split":cell["split"],"token_length":len(ids)-1,
                       "original_input_token_length":length,"context_truncated":length>config["max_tokens"],
                       "text_sha256":hashlib.sha256(text.encode()).hexdigest()}
                records.append((ids,row))
        if args.smoke_rows:
            # Include short and long actual texts, not just the first human rows.
            records = sorted(records, key=lambda r:len(r[0]))
            positions = np.linspace(0,len(records)-1,args.smoke_rows,dtype=int)
            records = [records[i] for i in positions]
        records.sort(key=lambda r:len(r[0]))
        written, offset, parity = 0, 0, []
        partial = output.with_suffix(".partial.jsonl")
        if partial.exists():
            raise RuntimeError(f"partial output requires inspection before rerun: {partial}")
        with partial.open("x") as handle:
            while offset < len(records):
                stop = min(offset + max_batch, len(records))
                while stop > offset+1 and max(len(r[0]) for r in records[offset:stop])*(stop-offset)>max_budget:
                    stop -= 1
                batch = records[offset:stop]
                encoded = tokenizers[0].pad([{"input_ids":ids,"attention_mask":[1]*len(ids)} for ids,_ in batch],
                                             return_tensors="pt")
                tick = time.perf_counter()
                with torch.inference_mode():
                    lr = qm(**{k:v.to(qdevice) for k,v in encoded.items()},use_cache=False).logits
                    ls = lr if same else pm(**{k:v.to(pdevice) for k,v in encoded.items()},use_cache=False).logits
                torch.cuda.synchronize(qdevice); torch.cuda.synchronize(pdevice)
                report["forward_seconds"] += time.perf_counter()-tick
                tick = time.perf_counter()
                for position,(ids,row) in enumerate(batch):
                    n = len(ids)-1
                    labels = encoded["input_ids"][position,1:len(ids)].to(qdevice)
                    r, s = lr[position,:n], ls[position,:n]
                    feats = features(r,s,labels,worker["temperatures"],worker["masses"],worker.get("sensitivity",False))
                    if written == 0 or args.smoke_rows:
                        native = get_sampling_discrepancy_analytic(r[None].float(),s[None].to(qdevice).float(),labels[None])
                        exact = compute_official_module_features_token_chunked(r[None],s[None].to(qdevice),labels[None],
                                                                                OfficialModuleConfig(),token_chunk_size=128)
                        errors = [abs(native-feats["fastdetect_native"]),abs(exact["nucleus_entropy"]-feats[tag(1.2,.8,"entropy")])]
                        assert errors[0] < 1e-4 and errors[1] < 1e-4, errors
                        report["max_official_parity_error"] = max(report["max_official_parity_error"],errors[0])
                        report["max_s3_parity_error"] = max(report["max_s3_parity_error"],errors[1])
                        parity.append({"row_id":row["row_id"],"native_error":errors[0],"s3_error":errors[1]})
                    handle.write(json.dumps({**row,**feats},separators=(",",":"))+"\n")
                    written += 1
                report["module_seconds"] += time.perf_counter()-tick
                del lr,ls
                offset=stop
                if written % 200 < max_batch:
                    handle.flush()
                    elapsed=time.perf_counter()-start_time
                    print(json.dumps({"cell":cell["cell"],"cell_rows":written,"total_rows":report["rows"]+written,
                                      "elapsed":round(elapsed,1)}),flush=True)
        os.replace(partial,output)
        atomic_json(manifest,{"config_sha256":config_hash,"sha256":sha(output),"rows":written,
                              "input_sha256":cell["sha256"],"parity_checks":parity})
        report["rows"]+=written;report["cells"]+=1
        report["elapsed_seconds"]=time.perf_counter()-start_time
        atomic_json(groups/f"worker_{args.worker}_progress.json",report)
        if args.smoke_rows: break
    report["status"]="COMPLETE"
    report["elapsed_seconds"]=time.perf_counter()-start_time
    report["gpu_peak_allocated_bytes"]={str(i):torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())}
    atomic_json(groups/f"worker_{args.worker}_progress.json",report)
    print(json.dumps(report),flush=True)


if __name__ == "__main__": main()
