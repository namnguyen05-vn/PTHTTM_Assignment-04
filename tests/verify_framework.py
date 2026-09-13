"""Compare logits, input gradients and first-convolution gradients with NumPy.

Each framework runs in its own CPU process; dropout is disabled for deterministic
comparison. Normal training keeps dropout enabled in the improved architecture.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['OMP_NUM_THREADS']='4'
import argparse,json
import numpy as np
from src.numpy_cnn import CNN,Dropout,cross_entropy
from src.data import DATASETS,ROOT

def verify(backend):
    results=[]
    for name,cfg in DATASETS.items():
        for variant in ['baseline','improved']:
            net=CNN(cfg['channels'],cfg['size'],cfg['classes'],variant)
            for layer in net.layers:
                if isinstance(layer,Dropout):layer.rate=0
            state=net.state()
            x=np.random.default_rng(901).normal(size=(3,cfg['channels'],cfg['size'],cfg['size'])).astype('float32')
            y=np.array([0,1,2],dtype='int64')
            ref=net.forward(x,True);loss,g=cross_entropy(ref,y)
            dx=net.backward(g);dw=net.layers[0].dw
            if backend=='pytorch':
                import torch
                from src.torch_cnn import TorchCNN
                torch.set_num_threads(4)
                model=TorchCNN(cfg['channels'],cfg['size'],cfg['classes'],variant);model.load_numpy(state)
                for layer in model.modules():
                    if isinstance(layer,torch.nn.Dropout):layer.p=0
                xx=torch.tensor(x,requires_grad=True);logits=model(xx)
                ll=torch.nn.functional.cross_entropy(logits,torch.tensor(y));ll.backward()
                actual=logits.detach().numpy();actual_dx=xx.grad.numpy();actual_dw=model.layers[0].weight.grad.numpy()
                model.eval();infer=model(torch.tensor(x)).detach().numpy()
            else:
                import tensorflow as tf
                from src.tf_cnn import build_model,load_numpy
                model=build_model(cfg['channels'],cfg['size'],cfg['classes'],variant);load_numpy(model,state,variant)
                for layer in model.layers:
                    if isinstance(layer,tf.keras.layers.Dropout):layer.rate=0
                xx=tf.Variable(x.transpose(0,2,3,1))
                with tf.GradientTape() as tape:
                    logits=model(xx,training=True)
                    ll=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits))
                actual_dx,actual_dw=tape.gradient(ll,[xx,model.get_layer('conv1').kernel])
                actual=logits.numpy();actual_dx=actual_dx.numpy().transpose(0,3,1,2);actual_dw=actual_dw.numpy().transpose(3,2,0,1)
                infer=model(xx,training=False).numpy()
            errors={}
            for key,a,b in [('train_logits',actual,ref),('input_gradient',actual_dx,dx),('conv1_gradient',actual_dw,dw),('eval_logits',infer,net.forward(x,False))]:
                errors[key]=float(np.max(np.abs(a-b)))
                np.testing.assert_allclose(a,b,atol=2e-4,rtol=3e-4,err_msg=f'{backend}/{name}/{variant}/{key}')
            results.append(dict(backend=backend,dataset=name,variant=variant,passed=True,max_absolute_errors=errors))
    (ROOT/'results'/f'verification_{backend}.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=['pytorch','tensorflow'],required=True)
    verify(p.parse_args().backend)
