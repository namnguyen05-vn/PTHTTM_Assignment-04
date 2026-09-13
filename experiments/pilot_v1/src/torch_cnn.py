"""PyTorch equivalent of the NumPy CNN (including population-variance BatchNorm)."""
import torch
from torch import nn

class PopulationBatchNorm(nn.Module):
    """Match NumPy/TF running variance; PyTorch's standard layer uses an unbiased estimate."""
    def __init__(self,c):
        super().__init__();self.weight=nn.Parameter(torch.ones(c));self.bias=nn.Parameter(torch.zeros(c))
        self.register_buffer('running_mean',torch.zeros(c));self.register_buffer('running_var',torch.ones(c))
    def forward(self,x):
        if self.training:
            mean=x.mean((0,2,3));var=x.var((0,2,3),unbiased=False)
            with torch.no_grad():
                self.running_mean.mul_(0.9).add_(mean.detach(),alpha=0.1)
                self.running_var.mul_(0.9).add_(var.detach(),alpha=0.1)
        else:mean,var=self.running_mean,self.running_var
        return (x-mean[None,:,None,None])*torch.rsqrt(var[None,:,None,None]+1e-5)*self.weight[None,:,None,None]+self.bias[None,:,None,None]

class Residual(nn.Module):
    def __init__(self,c):
        super().__init__();self.conv=nn.Conv2d(c,c,3,padding=1);self.bn=PopulationBatchNorm(c)
    def forward(self,x):return torch.relu(self.bn(self.conv(x))+x)

class TorchCNN(nn.Module):
    def __init__(self,channels,size,classes,variant='baseline'):
        super().__init__();improved=variant=='improved'
        layers=[nn.Conv2d(channels,8,3,padding=1)]
        if improved:layers.append(PopulationBatchNorm(8))
        layers += [nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(8,16,3,padding=1)]
        if improved:layers.append(PopulationBatchNorm(16))
        layers.append(nn.ReLU())
        if improved:layers.append(Residual(16))
        layers += [nn.MaxPool2d(2),nn.Flatten(),nn.Linear(16*(size//4)**2,64),nn.ReLU()]
        if improved:layers.append(nn.Dropout(0.25))
        layers.append(nn.Linear(64,classes));self.layers=nn.Sequential(*layers)
    def forward(self,x):return self.layers(x)
    def load_numpy(self,state):
        with torch.no_grad():
            for i,layer in enumerate(self.layers):
                prefix=f'{i}.'
                if isinstance(layer,(nn.Conv2d,nn.Linear)):
                    w=state[prefix+'weight'];w=w.T if isinstance(layer,nn.Linear) else w
                    layer.weight.copy_(torch.from_numpy(w.copy()));layer.bias.copy_(torch.from_numpy(state[prefix+'bias']))
                elif isinstance(layer,PopulationBatchNorm):
                    self._bn(layer,state,prefix)
                elif isinstance(layer,Residual):
                    layer.conv.weight.copy_(torch.from_numpy(state[prefix+'conv_weight']))
                    layer.conv.bias.copy_(torch.from_numpy(state[prefix+'conv_bias']))
                    self._bn(layer.bn,state,prefix+'bn_')
    @staticmethod
    def _bn(layer,state,p):
        for a,b in [('weight','gamma'),('bias','beta'),('running_mean','running_mean'),('running_var','running_var')]:
            getattr(layer,a).copy_(torch.from_numpy(state[p+b]))
