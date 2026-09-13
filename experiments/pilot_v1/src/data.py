"""Shared dataset definitions, Kaggle provenance and immutable train/val splits."""
import os,json,pickle,hashlib
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATASETS={
 'mnist':dict(channels=1,size=28,classes=10,kaggle='oddrationale/mnist-in-csv',epochs=5),
 'cifar10':dict(channels=3,size=32,classes=10,kaggle='pankrzysiu/cifar10-python',epochs=10),
 'cifar100':dict(channels=3,size=32,classes=100,kaggle='fedesoriano/cifar100',epochs=12),
}

def data_root():
    path=os.environ.get('CNN_DATA_DIR')
    if path:return Path(path)
    local=ROOT/'data'
    if local.exists() and (local/'mnist.npz').exists():return local
    windows=Path('E:/PTHTTM/ASG_04_data')
    return windows if windows.exists() else local

class ArrayUnpickler(pickle.Unpickler):
    """Allow only the NumPy constructors required by the CIFAR Python archives."""
    def find_class(self,module,name):
        allowed={('numpy.core.multiarray','_reconstruct'),('numpy._core.multiarray','_reconstruct'),('numpy','ndarray'),('numpy','dtype'),('_codecs','encode')}
        if (module,name) not in allowed:raise pickle.UnpicklingError(f'Unsupported object: {module}.{name}')
        return super().find_class(module,name)

def read_pickle(path):
    with open(path,'rb') as f:return ArrayUnpickler(f,encoding='bytes').load()

def prepare_data(root=None):
    root=Path(root or data_root());root.mkdir(parents=True,exist_ok=True)
    manifests=[]
    for name,cfg in DATASETS.items():
        dest=root/(name+'.npz')
        if not dest.exists():
            if name=='mnist':
                import pandas as pd
                def read_csv(split):
                    frame=pd.read_csv(root/name/f'mnist_{split}.csv')
                    y=frame.iloc[:,0].to_numpy(dtype='int64')
                    x=frame.iloc[:,1:].to_numpy(dtype='uint8').reshape(-1,1,28,28)
                    return x,y
                x,y=read_csv('train');xt,yt=read_csv('test');names=[str(i) for i in range(10)]
            elif name=='cifar10':
                p=root/name/'cifar-10-batches-py'
                batches=[read_pickle(p/f'data_batch_{i}') for i in range(1,6)]
                x=np.concatenate([b[b'data'] for b in batches]).reshape(-1,3,32,32)
                y=np.concatenate([b[b'labels'] for b in batches]).astype('int64')
                t=read_pickle(p/'test_batch');xt=t[b'data'].reshape(-1,3,32,32);yt=np.array(t[b'labels'],dtype='int64')
                names=[s.decode() for s in read_pickle(p/'batches.meta')[b'label_names']]
            else:
                p=root/name;b=read_pickle(p/'train');t=read_pickle(p/'test')
                x=b[b'data'].reshape(-1,3,32,32);y=np.array(b[b'fine_labels'],dtype='int64')
                xt=t[b'data'].reshape(-1,3,32,32);yt=np.array(t[b'fine_labels'],dtype='int64')
                names=[s.decode() for s in read_pickle(p/'meta')[b'fine_label_names']]
            rng=np.random.default_rng(42);train=[];val=[]
            for cls in range(cfg['classes']):
                ids=np.flatnonzero(y==cls);rng.shuffle(ids);nval=round(len(ids)*0.1)
                val.extend(ids[:nval]);train.extend(ids[nval:])
            train=np.array(train,dtype='int64');val=np.array(val,dtype='int64');rng.shuffle(train);rng.shuffle(val)
            np.savez_compressed(dest,x=x,y=y,x_test=xt,y_test=yt,train_ids=train,val_ids=val,class_names=np.array(names))
        with np.load(dest,allow_pickle=False) as d:
            assert set(d['train_ids']).isdisjoint(set(d['val_ids']))
            assert len(d['train_ids'])+len(d['val_ids'])==len(d['y'])
            assert d['x'].shape[1:]==(cfg['channels'],cfg['size'],cfg['size'])
            m=dict(dataset=name,train=len(d['train_ids']),validation=len(d['val_ids']),test=len(d['y_test']),classes=cfg['classes'],shape=list(d['x'].shape[1:]),train_class_counts=np.bincount(d['y'][d['train_ids']],minlength=cfg['classes']).tolist(),test_class_counts=np.bincount(d['y_test'],minlength=cfg['classes']).tolist(),split_sha256=hashlib.sha256(d['train_ids'].tobytes()+d['val_ids'].tobytes()).hexdigest(),kaggle_url='https://www.kaggle.com/datasets/'+cfg['kaggle'],prepared_sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
            manifests.append(m)
            sd=ROOT/'results'/'splits';sd.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(sd/(name+'.npz'),train_ids=d['train_ids'],validation_ids=d['val_ids'])
            print(json.dumps(m),flush=True)
    (ROOT/'results'/'dataset_manifest.json').write_text(json.dumps(manifests,indent=2),encoding='utf-8')
    return manifests

def load_data(name):
    path=data_root()/(name+'.npz')
    if not path.exists():raise FileNotFoundError(f'{path}: run the data preparation notebook first.')
    with np.load(path,allow_pickle=False) as d:
        return {k:d[k] for k in d.files}

def batches(x,y,ids,batch_size=128,seed=None):
    order=ids.copy()
    if seed is not None:np.random.default_rng(seed).shuffle(order)
    for start in range(0,len(order),batch_size):
        ix=order[start:start+batch_size]
        yield x[ix].astype('float32')/255.0,y[ix]

if __name__=='__main__':prepare_data()
