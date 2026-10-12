"""Hash the exact local checkpoint assets used by the supplement."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'scratch/baseline_audits'


def main():
    names=['opt-2.7B','falcon-7b','falcon-7b-instruct','stage3_watermarks/SemStamp-c4-sbert']
    models={}
    for name in names:
        path=Path('models')/name
        if (path/'model.safetensors.index.json').exists():
            weights=sorted(set(json.loads((path/'model.safetensors.index.json').read_text())['weight_map'].values()))
        elif (path/'model.safetensors').exists():weights=['model.safetensors']
        elif (path/'pytorch_model.bin.index.json').exists():weights=sorted(set(json.loads((path/'pytorch_model.bin.index.json').read_text())['weight_map'].values()))
        else:weights=['pytorch_model.bin']
        assets=sorted(set(weights+[p.name for p in path.glob('*.json')]+[p.name for p in path.glob('*.model')]+[p.name for p in path.glob('*.txt') if p.name in ['merges.txt','vocab.txt']]))
        files=[]
        for asset in assets:
            p=path/asset;before=p.stat();h=hashlib.sha256()
            with p.open('rb') as f:
                for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
            after=p.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
            files.append({'name':asset,'bytes':after.st_size,'sha256':h.hexdigest(),'mtime_ns':after.st_mtime_ns})
        models[name]={'path':str(path),'files':files}
        print(json.dumps({'checkpoint':name,'bytes':sum(f['bytes'] for f in files),'files':len(files)}),flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'checkpoint_hashes.json').write_text(json.dumps({'status':'COMPLETE','models':models},indent=2)+'\n')


if __name__=='__main__':main()
