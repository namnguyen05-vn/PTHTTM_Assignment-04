"""TensorFlow/Keras equivalent. Inputs are NHWC; flatten is explicitly NCHW."""
import tensorflow as tf
from tensorflow.keras import layers

def build_model(channels,size,classes,variant='baseline'):
    improved=variant=='improved';x0=tf.keras.Input((size,size,channels));x=x0
    x=layers.Conv2D(8,3,padding='same',name='conv1')(x)
    if improved:x=layers.BatchNormalization(momentum=0.9,epsilon=1e-5,fused=False,name='bn1')(x)
    x=layers.ReLU()(x);x=layers.MaxPool2D()(x)
    x=layers.Conv2D(16,3,padding='same',name='conv2')(x)
    if improved:x=layers.BatchNormalization(momentum=0.9,epsilon=1e-5,fused=False,name='bn2')(x)
    x=layers.ReLU()(x)
    if improved:
        residual=x;x=layers.Conv2D(16,3,padding='same',name='res_conv')(x)
        x=layers.BatchNormalization(momentum=0.9,epsilon=1e-5,fused=False,name='res_bn')(x)
        x=layers.ReLU()(layers.Add()([x,residual]))
    x=layers.MaxPool2D()(x)
    # Match the flattened feature order used by NumPy and PyTorch.
    x=layers.Permute((3,1,2))(x);x=layers.Reshape((16*(size//4)**2,))(x)
    x=layers.Dense(64,name='fc1')(x)
    if improved:x=layers.BatchNormalization(momentum=0.9,epsilon=1e-5,fused=False,name='bn_fc')(x)
    x=layers.ReLU()(x)
    if improved:x=layers.Dropout(0.25)(x)
    out=layers.Dense(classes,name='fc2')(x)
    return tf.keras.Model(x0,out,name='CNN_'+variant)

def load_numpy(model,state,variant):
    if variant=='baseline':mapping=[('conv1',0,'conv'),('conv2',3,'conv'),('fc1',7,'dense'),('fc2',9,'dense')]
    else:mapping=[('conv1',0,'conv'),('bn1',1,'bn'),('conv2',4,'conv'),('bn2',5,'bn'),('res_conv',7,'res_conv'),('res_bn',7,'res_bn'),('fc1',10,'dense'),('bn_fc',11,'bn'),('fc2',14,'dense')]
    for name,i,kind in mapping:
        prefix=f'{i}.'
        if kind=='res_conv':prefix+='conv_';kind='conv'
        if kind=='res_bn':prefix+='bn_';kind='bn'
        if kind=='conv':weights=[state[prefix+'weight'].transpose(2,3,1,0),state[prefix+'bias']]
        elif kind=='dense':weights=[state[prefix+'weight'],state[prefix+'bias']]
        else:weights=[state[prefix+k] for k in ['gamma','beta','running_mean','running_var']]
        model.get_layer(name).set_weights(weights)
