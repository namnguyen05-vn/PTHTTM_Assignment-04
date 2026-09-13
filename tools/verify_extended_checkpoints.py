"""Each checkpoint is reloaded in a fresh CPU process; cache verified records."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from run_extended import jobs
from src.train import experiment_dir

out=ROOT/'results/extended_checkpoint_verification.json'
records=json.loads(out.read_text()) if out.exists() else []
fields=['dataset','backend','variant','seed','ablation']
def key(d):return tuple(d.get(k) for k in fields)
done={key(d):d for d in records}
env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='-1';env['TF_CPP_MIN_LOG_LEVEL']='3';env['PYTHONUTF8']='1'
for job in jobs():
    folder=experiment_dir(**job)
    model_file={'numpy':'numpy_cnn.py','pytorch':'torch_cnn.py','tensorflow':'tf_cnn.py'}[job['backend']]
    digest=hashlib.sha256()
    for file in [next(folder.glob('weights.*')),folder/'test_outputs.npz',folder/'config.json',
                 ROOT/'src'/model_file,ROOT/'src/predict.py']:
        digest.update(file.name.encode());digest.update(file.read_bytes())
    fingerprint=digest.hexdigest()
    if key(job) in done and done[key(job)].get('verification_fingerprint')==fingerprint:continue
    cmd=[sys.executable,'-m','src.predict','--verify-checkpoint']
    for k,v in job.items():
        if v is not None:cmd.extend(['--'+k,str(v)])
    result=subprocess.run(cmd,cwd=ROOT,env=env,check=True,capture_output=True,text=True,encoding='utf-8')
    record=json.loads(result.stdout.strip().splitlines()[-1]);assert record['checkpoint_verified']
    record['verification_fingerprint']=fingerprint
    records=[r for r in records if key(r)!=key(job)]
    records.append(record);out.write_text(json.dumps(records,indent=2),encoding='utf-8')
    print('Verified',len(records),'/72',job,flush=True)
assert len(records)==72
