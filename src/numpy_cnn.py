"""CNN from scratch. All forward/backward operations and Adam updates use NumPy.

Layout is NCHW. Convolution implements cross-correlation (as in both frameworks).
No autograd, torch or TensorFlow is used in this module.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class Layer:
    def params(self): return []
    def state(self): return {}


class Conv2D(Layer):
    def __init__(self, cin, cout, rng, kernel=3, padding=1):
        self.k, self.p = kernel, padding
        self.w = (rng.standard_normal((cout, cin, kernel, kernel)) * np.sqrt(2/(cin*kernel*kernel))).astype('float32')
        self.b = np.zeros(cout, dtype='float32')
    def forward(self, x, training=True):
        n,c,h,w = x.shape
        xp = np.pad(x, ((0,0),(0,0),(self.p,self.p),(self.p,self.p)))
        win = sliding_window_view(xp, (self.k,self.k), axis=(2,3))
        oh,ow = win.shape[2:4]
        cols = win.transpose(0,2,3,1,4,5).reshape(n*oh*ow,-1)
        if training: self.cache = (x.shape, cols, oh, ow)
        y = cols @ self.w.reshape(len(self.w),-1).T + self.b
        return y.reshape(n,oh,ow,-1).transpose(0,3,1,2)
    def backward(self, dy):
        shape,cols,oh,ow = self.cache
        n,c,h,w = shape
        g = dy.transpose(0,2,3,1).reshape(-1,len(self.w))
        self.dw = (g.T @ cols).reshape(self.w.shape)
        self.db = g.sum(axis=0)
        dc = (g @ self.w.reshape(len(self.w),-1)).reshape(n,oh,ow,c,self.k,self.k)
        dxp = np.zeros((n,c,h+2*self.p,w+2*self.p),dtype='float32')
        for u in range(self.k):
            for v in range(self.k):
                dxp[:,:,u:u+oh,v:v+ow] += dc[:,:,:,:,u,v].transpose(0,3,1,2)
        return dxp[:,:,self.p:self.p+h,self.p:self.p+w]
    def params(self): return [(self.w,self.dw),(self.b,self.db)]
    def state(self): return {'weight':self.w, 'bias':self.b}


class BatchNorm2D(Layer):
    def __init__(self, channels, eps=1e-5, momentum=0.1):
        self.gamma=np.ones(channels,dtype='float32');self.beta=np.zeros(channels,dtype='float32')
        self.mean=np.zeros(channels,dtype='float32');self.var=np.ones(channels,dtype='float32')
        self.eps,self.momentum=eps,momentum
    def forward(self,x,training=True):
        axes=(0,2,3)
        if training:
            mean=x.mean(axis=axes);var=x.var(axis=axes)
            # Population variance is used in all three implementations.
            self.mean *= 1-self.momentum;self.mean += self.momentum*mean
            self.var *= 1-self.momentum;self.var += self.momentum*var
        else: mean,var=self.mean,self.var
        inv=1/np.sqrt(var+self.eps)
        z=(x-mean[None,:,None,None])*inv[None,:,None,None]
        if training:self.cache=(z,inv)
        return z*self.gamma[None,:,None,None]+self.beta[None,:,None,None]
    def backward(self,g):
        z,inv=self.cache;axes=(0,2,3)
        self.dg=(g*z).sum(axis=axes);self.db=g.sum(axis=axes)
        dx=(g-g.mean(axis=axes,keepdims=True)-z*(g*z).mean(axis=axes,keepdims=True))
        return dx*(self.gamma*inv)[None,:,None,None]
    def params(self):return [(self.gamma,self.dg),(self.beta,self.db)]
    def state(self):return {'gamma':self.gamma,'beta':self.beta,'running_mean':self.mean,'running_var':self.var}


class ReLU(Layer):
    def forward(self,x,training=True):
        if training:self.mask=x>0
        return np.maximum(x,0)
    def backward(self,g):return g*self.mask


class BatchNorm1D(BatchNorm2D):
    """The same manual normalization over the batch for Dense features."""
    def forward(self,x,training=True):
        return super().forward(x[:,:,None,None],training)[:,:,0,0]
    def backward(self,g):
        return super().backward(g[:,:,None,None])[:,:,0,0]


class MaxPool2D(Layer):
    def forward(self,x,training=True):
        n,c,h,w=x.shape;oh,ow=h//2,w//2
        windows=x[:,:,:oh*2,:ow*2].reshape(n,c,oh,2,ow,2).transpose(0,1,2,4,3,5).reshape(n,c,oh,ow,4)
        if training:self.cache=(x.shape,windows.argmax(axis=-1))
        return windows.max(axis=-1)
    def backward(self,g):
        shape,idx=self.cache;n,c,h,w=shape;oh,ow=h//2,w//2
        out=np.zeros((*g.shape,4),dtype='float32')
        np.put_along_axis(out,idx[...,None],g[...,None],axis=-1)
        out=out.reshape(n,c,oh,ow,2,2).transpose(0,1,2,4,3,5).reshape(n,c,oh*2,ow*2)
        dx=np.zeros(shape,dtype='float32');dx[:,:,:oh*2,:ow*2]=out
        return dx


class Flatten(Layer):
    def forward(self,x,training=True):
        if training:self.shape=x.shape
        return x.reshape(len(x),-1)
    def backward(self,g):return g.reshape(self.shape)


class Dense(Layer):
    def __init__(self,cin,cout,rng):
        self.w=(rng.standard_normal((cin,cout))*np.sqrt(2/cin)).astype('float32')
        self.b=np.zeros(cout,dtype='float32')
    def forward(self,x,training=True):
        if training:self.x=x
        return x@self.w+self.b
    def backward(self,g):
        self.dw=self.x.T@g;self.db=g.sum(axis=0)
        return g@self.w.T
    def params(self):return [(self.w,self.dw),(self.b,self.db)]
    def state(self):return {'weight':self.w,'bias':self.b}


class Dropout(Layer):
    def __init__(self,rng,rate=0.25):self.rng,self.rate=rng,rate
    def forward(self,x,training=True):
        if not training:return x
        self.mask=(self.rng.random(x.shape)>=self.rate).astype('float32')/(1-self.rate)
        return x*self.mask
    def backward(self,g):return g*self.mask


class Residual(Layer):
    """ReLU(BN(Conv(x)) + x), preserving the shape and adding both gradients."""
    def __init__(self,channels,rng):
        self.conv=Conv2D(channels,channels,rng);self.bn=BatchNorm2D(channels);self.relu=ReLU()
    def forward(self,x,training=True):
        return self.relu.forward(self.bn.forward(self.conv.forward(x,training),training)+x,training)
    def backward(self,g):
        g=self.relu.backward(g)
        return g+self.conv.backward(self.bn.backward(g))
    def params(self):return self.conv.params()+self.bn.params()
    def state(self):return {**{'conv_'+k:v for k,v in self.conv.state().items()},**{'bn_'+k:v for k,v in self.bn.state().items()}}


class CNN:
    def __init__(self,channels,size,classes,variant='baseline',seed=42):
        rng=np.random.default_rng(seed);improved=variant=='improved'
        self.layers=[Conv2D(channels,8,rng)]
        if improved:self.layers.append(BatchNorm2D(8))
        self.layers += [ReLU(),MaxPool2D(),Conv2D(8,16,rng)]
        if improved:self.layers.append(BatchNorm2D(16))
        self.layers.append(ReLU())
        if improved:self.layers.append(Residual(16,rng))
        self.layers += [MaxPool2D(),Flatten(),Dense(16*(size//4)**2,64,rng)]
        if improved:self.layers.append(BatchNorm1D(64))
        self.layers.append(ReLU())
        if improved:self.layers.append(Dropout(rng))
        self.layers.append(Dense(64,classes,rng))
    def forward(self,x,training=True):
        for layer in self.layers:x=layer.forward(x,training)
        return x
    def backward(self,g):
        for layer in reversed(self.layers):g=layer.backward(g)
        return g
    def params(self):return [p for layer in self.layers for p in layer.params()]
    def state(self):return {f'{i}.{k}':v.copy() for i,layer in enumerate(self.layers) for k,v in layer.state().items()}
    def load_state(self,state):
        for i,layer in enumerate(self.layers):
            for k,v in layer.state().items():v[:]=state[f'{i}.{k}']
    def parameter_count(self):
        return sum(v.size for layer in self.layers for k,v in layer.state().items() if not k.endswith(('running_mean','running_var')))


def cross_entropy(logits,labels):
    z=logits-logits.max(axis=1,keepdims=True)
    logsum=np.log(np.exp(z).sum(axis=1,keepdims=True))
    loss=float((logsum[:,0]-z[np.arange(len(labels)),labels]).mean())
    grad=np.exp(z-logsum)
    grad[np.arange(len(labels)),labels]-=1
    return loss,grad/len(labels)


class Adam:
    def __init__(self,lr=1e-3,beta1=0.9,beta2=0.999,eps=1e-8):
        self.lr,self.b1,self.b2,self.eps=lr,beta1,beta2,eps;self.t=0;self.m=[];self.v=[]
    def step(self,params):
        if not self.m:
            self.m=[np.zeros_like(w) for w,g in params];self.v=[np.zeros_like(w) for w,g in params]
        self.t+=1
        for (w,g),m,v in zip(params,self.m,self.v):
            m*=self.b1;m+=(1-self.b1)*g
            v*=self.b2;v+=(1-self.b2)*g*g
            w-=self.lr*(m/(1-self.b1**self.t))/(np.sqrt(v/(1-self.b2**self.t))+self.eps)
