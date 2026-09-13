"""Executable synthetic example: image -> Conv -> ReLU -> Pool -> Dense -> CE."""
import json
from pathlib import Path
import numpy as np
from .numpy_cnn import Conv2D, ReLU, MaxPool2D, Flatten, Dense, Adam, cross_entropy
from .data import ROOT

def example():
    rng=np.random.default_rng(0)
    x=np.array([[1,0,2,3,1],[4,6,6,8,2],[3,1,1,0,2],
                [1,2,2,4,0],[0,1,3,1,2]],dtype='float32')[None,None]
    conv=Conv2D(1,1,rng,kernel=2,padding=0)
    conv.w[:]=[[[[1,0],[0,-1]]]];conv.b[:]=0
    dense=Dense(4,3,rng)
    dense.w[:]=[[.1,-.2,.0],[.0,.1,-.1],[-.1,.0,.2],[.2,-.1,.1]]
    dense.b[:]=[0,.1,-.1]
    layers=[conv,ReLU(),MaxPool2D(),Flatten(),dense]
    record={'purpose':'synthetic arithmetic illustration, not a dataset experiment',
            'image':x[0,0].tolist(),'kernel':conv.w[0,0].tolist(),
            'dense_weights':dense.w.tolist(),'dense_bias':dense.b.tolist(),'label':1}
    z=x
    for name,layer in zip(['convolution','relu','pool','flatten','logits'],layers):
        z=layer.forward(z,True);record[name]=z.squeeze().tolist()
    loss,g=cross_entropy(z,np.array([1]))
    record['loss']=loss;record['logit_gradient']=g[0].tolist()
    prob=np.exp(z-z.max());prob/=prob.sum();record['probabilities']=prob[0].tolist()
    for layer in reversed(layers):g=layer.backward(g)
    record['input_gradient']=g[0,0].tolist();record['kernel_gradient']=conv.dw[0,0].tolist()
    record['dense_gradient']=dense.dw.tolist()
    before=conv.w.copy();Adam(lr=.001).step([p for layer in layers for p in layer.params()])
    record['kernel_after_adam']=conv.w[0,0].tolist()
    assert np.isfinite(loss) and np.allclose(prob.sum(),1)
    # Independent direct loop verifies the vectorized convolution.
    direct=np.array([[float((x[0,0,i:i+2,j:j+2]*before[0,0]).sum())
                      for j in range(4)] for i in range(4)])
    np.testing.assert_allclose(direct,record['convolution'])
    return record

if __name__=='__main__':
    target=ROOT/'results/teaching_example.json'
    target.write_text(json.dumps(example(),indent=2),encoding='utf-8')
    print(target)
