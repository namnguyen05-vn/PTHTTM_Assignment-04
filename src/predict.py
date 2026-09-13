"""Load a submitted checkpoint in a fresh process and inspect test predictions."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
import argparse,json
import numpy as np
from .data import ROOT,DATASETS,load_data

def predict(backend,dataset,variant,images,seed=42,ablation=None):
    """images: NCHW uint8 pixels, or float32 already divided by 255. Uses CPU."""
    from .train import experiment_dir
    cfg=DATASETS[dataset];folder=experiment_dir(backend,dataset,variant,seed,ablation)
    x=images.astype('float32')/255 if images.dtype==np.uint8 else images.astype('float32')
    if backend=='numpy':
        from .numpy_cnn import CNN
        model=CNN(cfg['channels'],cfg['size'],cfg['classes'],variant)
        with np.load(folder/'weights.npz',allow_pickle=False) as weights:model.load_state(dict(weights))
        return model.forward(x,False)
    if backend=='pytorch':
        import torch
        from .torch_cnn import TorchCNN
        torch.set_num_threads(1)
        model=TorchCNN(cfg['channels'],cfg['size'],cfg['classes'],variant).ablate(ablation)
        model.load_state_dict(torch.load(folder/'weights.pt',map_location='cpu',weights_only=True));model.eval()
        with torch.no_grad():return model(torch.from_numpy(np.ascontiguousarray(x))).numpy()
    if backend=='tensorflow':
        os.environ['CUDA_VISIBLE_DEVICES']='-1'
        import tensorflow as tf
        from .tf_cnn import build_model
        model=build_model(cfg['channels'],cfg['size'],cfg['classes'],variant)
        model.load_weights(str(folder/'weights.h5'))
        return model(np.ascontiguousarray(x.transpose(0,2,3,1)),training=False).numpy()
    raise ValueError(backend)

def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=['numpy','pytorch','tensorflow'],required=True)
    p.add_argument('--dataset',choices=list(DATASETS),required=True);p.add_argument('--variant',choices=['baseline','improved'],default='improved')
    p.add_argument('--test-id',type=int,default=0);p.add_argument('--verify-checkpoint',action='store_true')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--ablation',choices=['no_bn','no_skip','no_dropout']);a=p.parse_args()
    d=load_data(a.dataset)
    ids=np.array([0,7,113,999,5012,9999]) if a.verify_checkpoint else np.array([a.test_id])
    if np.any((ids<0)|(ids>=len(d['y_test']))):raise ValueError('test-id must be in [0, 9999]')
    logits=predict(a.backend,a.dataset,a.variant,d['x_test'][ids],a.seed,a.ablation)
    if a.verify_checkpoint:
        from .train import experiment_dir
        with np.load(experiment_dir(a.backend,a.dataset,a.variant,a.seed,a.ablation)/'test_outputs.npz') as saved:reference=saved['logits'][ids]
        np.testing.assert_allclose(logits,reference,atol=3e-4,rtol=3e-4)
        error=float(np.max(np.abs(logits-reference)))
        print(json.dumps(dict(dataset=a.dataset,backend=a.backend,variant=a.variant,seed=a.seed,ablation=a.ablation,checkpoint_verified=True,samples=len(ids),max_abs_error=error)))
    else:
        z=logits[0]-logits[0].max();prob=np.exp(z);prob/=prob.sum();order=np.argsort(prob)[-5:][::-1]
        print(json.dumps(dict(dataset=a.dataset,test_id=a.test_id,true_label=str(d['class_names'][d['y_test'][a.test_id]]),top5=[dict(label=str(d['class_names'][i]),probability=float(prob[i])) for i in order]),indent=2))

if __name__=='__main__':main()
