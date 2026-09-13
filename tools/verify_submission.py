"""Final artifact integrity checks; does not replace visual PDF review."""
import hashlib,json,sys
from pathlib import Path
import nbformat
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from run_extended import jobs
from src.train import experiment_dir

def read(name):return json.loads((ROOT/'results'/name).read_text(encoding='utf-8'))
outputs=read('extended_output_verification.json');checkpoints=read('extended_checkpoint_verification.json')
assert len(outputs)==len(checkpoints)==72
assert all(x['metrics_predictions_confusion_best_epoch_verified'] for x in outputs)
assert all(x['checkpoint_verified'] for x in checkpoints)
expected=set()
for job in jobs():
    folder=experiment_dir(**job);expected.add(folder.relative_to(ROOT).as_posix())
    for name in ['config.json','history.csv','metrics.json','predictions.csv','confusion_matrix.csv','test_outputs.npz']:
        assert (folder/name).is_file(),folder/name
    assert len(list(folder.glob('weights.*')))==1
assert {x['folder'] for x in outputs}==expected
assert {(x['dataset'],x['backend'],x['variant'],x['seed'],x['ablation']) for x in checkpoints}=={
    (x['dataset'],x['backend'],x['variant'],x['seed'],x['ablation']) for x in jobs()}
notebooks=[]
for path in sorted((ROOT/'notebooks').glob('*.ipynb')):
    nb=nbformat.read(path,as_version=4);nbformat.validate(nb)
    cells=[c for c in nb.cells if c.cell_type=='code']
    assert all(c.execution_count is not None for c in cells)
    assert not any(o.output_type=='error' for c in cells for o in c.outputs)
    notebooks.append(dict(notebook=path.name,executed_code_cells=len(cells),errors=0))
assert notebooks==read('notebook_verification.json')
assert len(notebooks)==8 and sum(x['executed_code_cells'] for x in notebooks)==60
unit=read('unit_test_verification.json');assert unit['passed'] and unit['tests_run']==15
pdf=ROOT/'report/Assignment04_NguyenNgocHoangNam_B23DCCN585.pdf'
r=PdfReader(pdf);assert len(r.pages)==78
assert all(len(page.extract_text().strip())>40 for page in r.pages)
layout=read('report_layout.json');assert len(layout)==77
index=read('report_caption_index.json')
record=dict(experiments=72,main_runs=54,ablation_runs=18,excluded_pilot_runs=4,
            saved_metrics_verified=72,checkpoints_verified=72,executed_notebooks=8,
            executed_code_cells=60,notebook_errors=0,unit_tests_passed=15,report_pages=78,
            figures=len(index['figures']),tables=len(index['tables']),code_listings=len(index['code_listings']),
            report_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),status='passed')
(ROOT/'results/submission_verification.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record,indent=2))
