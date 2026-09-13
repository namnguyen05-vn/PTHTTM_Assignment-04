"""Execute notebooks and save real outputs; failures stop publication."""
import argparse,json,asyncio,os
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]
if os.name=='nt':asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
p=argparse.ArgumentParser();p.add_argument('--prefix',default='');args=p.parse_args()
records=[]
for path in sorted((ROOT/'notebooks').glob(args.prefix+'*.ipynb')):
    nb=nbformat.read(path,as_version=4)
    client=NotebookClient(nb,timeout=7200,kernel_name='assignment04',resources={'metadata':{'path':str(ROOT/'notebooks')}})
    client.execute();nbformat.write(nb,path)
    count=sum(c.cell_type=='code' for c in nb.cells)
    assert all(c.execution_count is not None for c in nb.cells if c.cell_type=='code')
    assert not any(o.output_type=='error' for c in nb.cells if c.cell_type=='code' for o in c.outputs)
    records.append(dict(notebook=path.name,executed_code_cells=count,errors=0))
    print('Executed',path.name,count,'code cells',flush=True)
(ROOT/'results'/('notebook_verification'+args.prefix+'.json')).write_text(json.dumps(records,indent=2))
