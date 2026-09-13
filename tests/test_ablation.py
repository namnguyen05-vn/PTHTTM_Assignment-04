"""Validate controlled interventions before any ablation training."""
import unittest
import torch
from src.numpy_cnn import CNN
from src.torch_cnn import TorchCNN, PopulationBatchNorm, Residual
from src.train import experiment_dir

class AblationTests(unittest.TestCase):
    def model(self, ablation=None):
        m=TorchCNN(3,32,100,'improved')
        m.load_numpy(CNN(3,32,100,'improved',7).state())
        return m.ablate(ablation)

    def test_shared_weights(self):
        original=self.model().state_dict()
        for a in ['no_bn','no_skip','no_dropout']:
            for k,v in self.model(a).state_dict().items():
                torch.testing.assert_close(v,original[k],rtol=0,atol=0)

    def test_no_bn_removes_all_four(self):
        m=self.model('no_bn')
        self.assertFalse(any(isinstance(x,PopulationBatchNorm) for x in m.modules()))
        self.assertEqual(sum(p.numel() for p in m.parameters()),76020-2*(8+16+16+64))
        self.assertEqual(sum(isinstance(x,torch.nn.Conv2d) for x in m.modules()),3)

    def test_no_skip_preserves_branch_and_gradient(self):
        m=self.model('no_skip')
        block=next(x for x in m.modules() if isinstance(x,Residual));block.eval()
        x=torch.randn(2,16,8,8,requires_grad=True)
        y=block(x);expected=torch.relu(block.bn(block.conv(x)))
        torch.testing.assert_close(y,expected)
        gx=torch.autograd.grad(y.sum(),x)[0]
        ge=torch.autograd.grad(expected.sum(),x)[0]
        torch.testing.assert_close(gx,ge)
        self.assertEqual(sum(p.numel() for p in m.parameters()),76020)

    def test_dropout_identity(self):
        layer=next(x for x in self.model('no_dropout').modules() if isinstance(x,torch.nn.Dropout))
        layer.train();x=torch.randn(4,64)
        torch.testing.assert_close(layer(x),x,rtol=0,atol=0)

    def test_output_and_training_gradient(self):
        for a in ['no_bn','no_skip','no_dropout']:
            m=self.model(a);y=m(torch.randn(2,3,32,32))
            self.assertEqual(tuple(y.shape),(2,100))
            torch.nn.functional.cross_entropy(y,torch.tensor([1,99])).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))

    def test_seed_and_ablation_paths_do_not_collide(self):
        paths={experiment_dir('pytorch','cifar100','improved',seed,a)
               for seed in [42,7,2026] for a in [None,'no_bn','no_skip','no_dropout']}
        self.assertEqual(len(paths),12)

if __name__=='__main__':unittest.main()
