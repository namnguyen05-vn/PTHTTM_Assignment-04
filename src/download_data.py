"""Download public Kaggle archives, record checksums, safely extract and prepare."""
import hashlib,json,zipfile,argparse
from pathlib import Path
from urllib.request import urlopen,Request
from .data import DATASETS,data_root,prepare_data,ROOT

def download(root=None):
    root=Path(root or data_root());root.mkdir(parents=True,exist_ok=True)
    sources=[]
    for name,cfg in DATASETS.items():
        dest=root/(name+'.zip');slug=cfg['kaggle']
        url='https://www.kaggle.com/api/v1/datasets/download/'+slug
        if not dest.exists():
            print('Downloading '+name,flush=True)
            temp=dest.with_suffix('.part')
            with urlopen(Request(url,headers={'User-Agent':'Assignment04'}),timeout=120) as r,temp.open('wb') as f:
                while True:
                    chunk=r.read(2**20)
                    if not chunk:break
                    f.write(chunk)
            temp.replace(dest)
        target=root/name
        with zipfile.ZipFile(dest) as archive:
            entries=archive.namelist()
            for entry in archive.infolist():
                if not (target/entry.filename).resolve().is_relative_to(target.resolve()):
                    raise ValueError('Archive entry leaves its extraction directory')
            archive.extractall(target)
        sha=hashlib.sha256()
        with dest.open('rb') as f:
            for chunk in iter(lambda:f.read(2**20),b''):sha.update(chunk)
        source=dict(dataset=name,kaggle_slug=slug,url='https://www.kaggle.com/datasets/'+slug,zip_sha256=sha.hexdigest(),bytes=dest.stat().st_size,files=entries)
        (root/(name+'_source.json')).write_text(json.dumps(source,indent=2))
        sources.append(source)
    (ROOT/'results').mkdir(exist_ok=True)
    (ROOT/'results'/'data_sources.json').write_text(json.dumps(sources,indent=2))
    return prepare_data(root)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir')
    download(p.parse_args().data_dir)

