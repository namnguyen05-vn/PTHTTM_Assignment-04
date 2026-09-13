"""Independent finite-difference checks for the handwritten backward passes."""
import unittest
import numpy as np
from src.numpy_cnn import Conv2D,Dense,BatchNorm2D,BatchNorm1D,MaxPool2D,Residual,CNN,cross_entropy,Adam

class Gradients(unittest.TestCase):
    def numerical(self,layer,x,attribute=None,checks=8):
        rng=np.random.default_rng(7);y=layer.forward(x);g=rng.normal(size=y.shape).astype('float32')
        dx=layer.backward(g)
        a=x if attribute is None else getattr(layer,attribute)
        analytic=dx if attribute is None else getattr(layer,{'w':'dw','b':'db','gamma':'dg','beta':'db'}[attribute])
        analytic=analytic.copy()
        for flat in rng.choice(a.size,min(checks,a.size),replace=False):
            ix=np.unravel_index(flat,a.shape);old=a[ix].copy();eps=0.002
            a[ix]=old+eps;plus=float((layer.forward(x)*g).sum())
            a[ix]=old-eps;minus=float((layer.forward(x)*g).sum());a[ix]=old
            numeric=(plus-minus)/(2*eps)
            np.testing.assert_allclose(analytic[ix],numeric,atol=0.006,rtol=0.02)
    def test_conv_gradients(self):
        rng=np.random.default_rng(9);x=rng.normal(size=(2,2,4,4)).astype('float32');layer=Conv2D(2,3,rng)
        for name in [None,'w','b']:self.numerical(layer,x,name)
    def test_dense_gradients(self):
        rng=np.random.default_rng(9);x=rng.normal(size=(3,5)).astype('float32');layer=Dense(5,4,rng)
        for name in [None,'w','b']:self.numerical(layer,x,name)
    def test_bn_gradients(self):
        x=np.random.default_rng(9).normal(size=(2,3,4,4)).astype('float32');layer=BatchNorm2D(3)
        for name in [None,'gamma','beta']:self.numerical(layer,x,name)
    def test_pool_gradients(self):
        x=np.random.default_rng(8).normal(size=(2,2,4,4)).astype('float32');self.numerical(MaxPool2D(),x)
    def test_bn_dense_gradients(self):
        x=np.random.default_rng(9).normal(size=(6,5)).astype('float32');layer=BatchNorm1D(5)
        for name in [None,'gamma','beta']:self.numerical(layer,x,name)
    def test_residual_gradient(self):
        rng=np.random.default_rng(14);x=rng.normal(size=(2,2,4,4)).astype('float32');self.numerical(Residual(2,rng),x)
    def test_loss_stability(self):
        loss,g=cross_entropy(np.array([[1000,1001,999]],dtype='float32'),np.array([1]))
        self.assertTrue(np.isfinite(loss));self.assertAlmostEqual(float(g.sum()),0,places=6)
    def test_shapes_and_counts(self):
        for variant in ['baseline','improved']:
            m=CNN(3,32,100,variant);x=np.zeros((2,3,32,32),dtype='float32');y=m.forward(x)
            self.assertEqual(y.shape,(2,100));self.assertEqual(m.backward(np.ones_like(y)).shape,x.shape)
            self.assertEqual(m.parameter_count(),sum(w.size for w,g in m.params()))
    def test_learning_tiny_problem(self):
        rng=np.random.default_rng(1);x=rng.normal(size=(12,1,8,8)).astype('float32');y=np.arange(12)%3
        model=CNN(1,8,3);opt=Adam(lr=0.003);start=cross_entropy(model.forward(x),y)[0]
        for _ in range(80):
            loss,g=cross_entropy(model.forward(x),y);model.backward(g);opt.step(model.params())
        end=cross_entropy(model.forward(x,False),y)[0]
        self.assertLess(end,start*0.1)

if __name__=='__main__':unittest.main(verbosity=2)
