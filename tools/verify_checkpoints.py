import json,subprocess,sys,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];records=[]
p=argparse.ArgumentParser();p.add_argument('--only-new',action='store_true');args=p.parse_args()
previous=ROOT/'results'/'checkpoint_verification.json'
if args.only_new and previous.exists():records=json.loads(previous.read_text())
done={(r['dataset'],r['backend'],r['variant']) for r in records}
for path in sorted((ROOT/'results').glob('*/metrics.json')):
    m=json.loads(path.read_text())
    if (m['dataset'],m['backend'],m['variant']) in done:continue
    result=subprocess.run([sys.executable,'-m','src.predict','--backend',m['backend'],'--dataset',m['dataset'],'--variant',m['variant'],'--verify-checkpoint'],cwd=ROOT,capture_output=True,text=True,check=True)
    record=json.loads(result.stdout);records.append(record);print(json.dumps(record),flush=True)
(ROOT/'results'/'checkpoint_verification.json').write_text(json.dumps(records,indent=2))
