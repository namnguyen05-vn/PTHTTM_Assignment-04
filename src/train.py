"""Reproducible full-dataset training. Run: python -m src.train --help."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ.setdefault(key,'1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
import json,csv,time,platform,argparse,sys,copy
from pathlib import Path
import numpy as np
from .data import ROOT,DATASETS,load_data,batches
from .numpy_cnn import CNN,cross_entropy,Adam

def classification_metrics(y,logits,k):
    pred=logits.argmax(axis=1);cm=np.bincount(y*k+pred,minlength=k*k).reshape(k,k)
    tp=np.diag(cm).astype(float)
    precision=np.divide(tp,cm.sum(0),out=np.zeros(k),where=cm.sum(0)>0)
    recall=np.divide(tp,cm.sum(1),out=np.zeros(k),where=cm.sum(1)>0)
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(k),where=(precision+recall)>0)
    result={'accuracy':float((pred==y).mean()),'macro_precision':float(precision.mean()),'macro_recall':float(recall.mean()),'macro_f1':float(f1.mean()),'test_loss':cross_entropy(logits,y)[0]}
    if k>=5:result['top5_accuracy']=float(np.any(np.argsort(logits,axis=1)[:,-5:]==y[:,None],axis=1).mean())
    return result,cm

def experiment_dir(backend,dataset,variant,seed=42,ablation=None):
    name=f'{dataset}_{backend}_{variant}'
    if ablation:return ROOT/'results'/'ablation'/f'seed_{seed}'/(name+'_'+ablation)
    if seed!=42:return ROOT/'results'/'multiseed'/f'seed_{seed}'/name
    return ROOT/'results'/name

def _train(backend,dataset,variant,epochs=None,batch_size=128,seed=42,force=False,ablation=None):
    if ablation and (backend!='pytorch' or variant!='improved'):
        raise ValueError('Ablation requires the PyTorch improved architecture.')
    cfg=DATASETS[dataset];epochs=epochs or cfg['epochs']
    out=experiment_dir(backend,dataset,variant,seed,ablation);out.mkdir(parents=True,exist_ok=True)
    if (out/'metrics.json').exists() and not force:
        saved=json.loads((out/'config.json').read_text())
        for key,value in dict(seed=seed,epochs=epochs,batch_size=batch_size,ablation=ablation).items():
            if saved.get(key)!=value:raise ValueError(f'Existing {out}: {key} differs; use --force or another seed.')
        return json.loads((out/'metrics.json').read_text())
    data=load_data(dataset);x,y=data['x'],data['y'];ti,vi=data['train_ids'],data['val_ids']
    spec=dict(channels=cfg['channels'],size=cfg['size'],classes=cfg['classes'],variant=variant,seed=seed)
    initial=CNN(**spec);state=initial.state();params=initial.parameter_count()
    np.random.seed(seed)
    if backend=='numpy':
        model=initial;optimizer=Adam();device='CPU';version=np.__version__
        def predict(b):return model.forward(b,False)
        def step(b,t):
            logits=model.forward(b,True);loss,g=cross_entropy(logits,t);model.backward(g);optimizer.step(model.params())
            return loss,int((logits.argmax(1)==t).sum())
        def save():np.savez_compressed(out/'weights.npz',**model.state())
        def restore():
            with np.load(out/'weights.npz') as d:model.load_state(dict(d))
    elif backend=='pytorch':
        import torch
        from .torch_cnn import TorchCNN
        torch.manual_seed(seed);torch.set_num_threads(4)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        device='cuda' if torch.cuda.is_available() else 'cpu';version=torch.__version__
        model=TorchCNN(cfg['channels'],cfg['size'],cfg['classes'],variant);model.load_numpy(state);model.to(device)
        assert sum(p.numel() for p in model.parameters())==params
        model.ablate(ablation);params=sum(p.numel() for p in model.parameters())
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-3,eps=1e-8)
        def predict(b):
            model.eval()
            with torch.no_grad():return model(torch.from_numpy(np.ascontiguousarray(b)).to(device)).cpu().numpy()
        def step(b,t):
            model.train();optimizer.zero_grad(set_to_none=True)
            logits=model(torch.from_numpy(np.ascontiguousarray(b)).to(device));target=torch.from_numpy(t).to(device)
            loss=torch.nn.functional.cross_entropy(logits,target);loss.backward();optimizer.step()
            return loss.item(),int((logits.argmax(1)==target).sum().item())
        def save():torch.save(model.state_dict(),out/'weights.pt')
        def restore():model.load_state_dict(torch.load(out/'weights.pt',map_location=device,weights_only=True))
    elif backend=='tensorflow':
        cuda=os.environ.get('CNN_CUDA_DIR','E:/PTHTTM/ASG_04_runtime/cuda/Library/bin')
        if os.name=='nt' and Path(cuda).is_dir():
            os.environ['PATH']=cuda+os.pathsep+os.environ['PATH'];dll=os.add_dll_directory(cuda)
        import tensorflow as tf
        from .tf_cnn import build_model,load_numpy
        tf.keras.utils.set_random_seed(seed)
        tf.config.threading.set_intra_op_parallelism_threads(4);tf.config.threading.set_inter_op_parallelism_threads(2)
        tf.config.experimental.enable_tensor_float_32_execution(False)
        gpus=tf.config.list_physical_devices('GPU')
        for gpu in gpus:tf.config.experimental.set_memory_growth(gpu,True)
        device='GPU' if gpus else 'CPU';version=tf.__version__
        model=build_model(cfg['channels'],cfg['size'],cfg['classes'],variant);load_numpy(model,state,variant)
        assert sum(int(np.prod(v.shape)) for v in model.trainable_weights)==params
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3,epsilon=1e-8)
        @tf.function(reduce_retracing=True)
        def train_step(b,t):
            with tf.GradientTape() as tape:
                logits=model(b,training=True)
                loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=t,logits=logits))
            optimizer.apply_gradients(zip(tape.gradient(loss,model.trainable_weights),model.trainable_weights))
            return loss,tf.reduce_sum(tf.cast(tf.argmax(logits,axis=1)==t,tf.int32))
        @tf.function(reduce_retracing=True)
        def infer(b):return model(b,training=False)
        def predict(b):return infer(np.ascontiguousarray(b.transpose(0,2,3,1))).numpy()
        def step(b,t):
            loss,correct=train_step(np.ascontiguousarray(b.transpose(0,2,3,1)),t)
            return float(loss.numpy()),int(correct.numpy())
        def save():model.save_weights(str(out/'weights.h5'))
        def restore():model.load_weights(str(out/'weights.h5'))
    else:raise ValueError(backend)
    history=[];best=float('inf');best_epoch=0;train_seconds=0;val_seconds=0
    print(f'START {dataset} {backend} {variant} seed={seed} ablation={ablation}: train={len(ti)} val={len(vi)} epochs={epochs} device={device} params={params}',flush=True)
    config=dict(dataset=dataset,backend=backend,variant=variant,epochs=epochs,batch_size=batch_size,seed=seed,learning_rate=0.001,optimizer='Adam',adam_beta1=0.9,adam_beta2=0.999,adam_epsilon=1e-8,normalization='uint8 / 255',train_samples=len(ti),validation_samples=len(vi),test_samples=len(data['y_test']),device=device,framework_version=version,python=sys.version,platform=platform.platform(),parameter_count=params,selection='minimum validation cross-entropy',augmentation=False,training_scope='all predefined training samples, no subsampling')
    from threadpoolctl import threadpool_info
    config['ablation']=ablation
    config['split_seed']=42
    config['cpu_thread_pools']=threadpool_info()
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    for epoch in range(1,epochs+1):
        ts=time.perf_counter();ls=0;correct=0
        for b,t in batches(x,y,ti,batch_size,seed+epoch):
            loss,c=step(b,t);ls+=loss*len(t);correct+=c
        elapsed=time.perf_counter()-ts;train_seconds+=elapsed;ts=time.perf_counter()
        logits=np.concatenate([predict(b) for b,t in batches(x,y,vi,batch_size)])
        vl=cross_entropy(logits,y[vi])[0];va=float((logits.argmax(1)==y[vi]).mean());ve=time.perf_counter()-ts;val_seconds+=ve
        if not np.isfinite(ls+vl):raise FloatingPointError('Non-finite loss. Stop rather than record invalid results.')
        if vl<best:best=vl;best_epoch=epoch;save()
        row=dict(epoch=epoch,train_loss=ls/len(ti),train_accuracy=correct/len(ti),val_loss=vl,val_accuracy=va,train_seconds=elapsed,validation_seconds=ve)
        history.append(row)
        with (out/'history.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerows(history)
        print(f'{dataset}/{backend}/{variant} epoch {epoch}/{epochs}: loss {row["train_loss"]:.4f} val_loss {vl:.4f} val_acc {va:.4f} train_s {elapsed:.1f}',flush=True)
    restore();xt,yt=data['x_test'],data['y_test'];ts=time.perf_counter()
    logits=np.concatenate([predict(b) for b,t in batches(xt,yt,np.arange(len(yt)),batch_size)])
    infer_seconds=time.perf_counter()-ts
    metrics,cm=classification_metrics(yt,logits,cfg['classes'])
    metrics.update({k:config[k] for k in ['dataset','backend','variant','parameter_count','train_samples','validation_samples','test_samples','device']})
    metrics.update(seed=seed,ablation=ablation,epochs=epochs,best_epoch=best_epoch,best_val_loss=best,train_seconds=train_seconds,validation_seconds=val_seconds,test_seconds=infer_seconds)
    shifted=logits-logits.max(1,keepdims=True);probs=np.exp(shifted);probs/=probs.sum(1,keepdims=True)
    with (out/'predictions.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['test_id','true_label','predicted_label','confidence'])
        writer.writerows(zip(range(len(yt)),yt.tolist(),logits.argmax(1).tolist(),probs.max(1).tolist()))
    np.savez_compressed(out/'test_outputs.npz',logits=logits,labels=yt)
    np.savetxt(out/'confusion_matrix.csv',cm,delimiter=',',fmt='%d')
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print('COMPLETE '+json.dumps(metrics),flush=True)
    return metrics

def train(backend,dataset,variant,epochs=None,batch_size=128,seed=42,force=False,ablation=None):
    # A notebook and the CLI may request the same configuration concurrently.
    # Serialize that configuration; after waiting, reuse its complete results.
    from filelock import FileLock
    folder=experiment_dir(backend,dataset,variant,seed,ablation)
    folder.mkdir(parents=True,exist_ok=True)
    with FileLock(str(folder/'.train.lock'),timeout=7200):
        return _train(backend,dataset,variant,epochs,batch_size,seed,force,ablation)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=['numpy','pytorch','tensorflow'],required=True)
    p.add_argument('--dataset',choices=list(DATASETS),required=True);p.add_argument('--variant',choices=['baseline','improved'],default='baseline')
    p.add_argument('--epochs',type=int);p.add_argument('--batch-size',type=int,default=128);p.add_argument('--seed',type=int,default=42);p.add_argument('--force',action='store_true')
    p.add_argument('--ablation',choices=['no_bn','no_skip','no_dropout'])
    train(**vars(p.parse_args()))
