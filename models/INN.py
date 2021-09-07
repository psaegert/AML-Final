import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class Permute(nn.Module):
    def __init__(self, features, device=None, dtype=None):
        super(Permute, self).__init__()

        factory_kwargs = {'device': device, 'dtype': dtype}
        self.features = features

        self.Q = Parameter(torch.zeros((self.features, self.features), **factory_kwargs))

        self.reset_parameters()

    def reset_parameters(self):
        self.Q[torch.randperm(self.features)] = torch.ones(self.features)

    def forward(self, X):
        return X.matmul(self.Q)

    def reverse(self, Z):
        return Z.matmul(self.Q.t())


class AffineCoupling(nn.Module):
    def __init__(self, features, device=None, dtype=None):
        super(AffineCoupling, self).__init__()

        self.features = features

        self.split_size_A = features // 2
        self.split_size_B = self.features - self.split_size_A

        self.S = ... # id, nn, or something else...
        self.T = ...


    def forward(self, input):
        Xa, Xb = torch.chunk(input, 2, dim=1)
        Zb = self.S(Xa) * Xb + self.T(Xa)

        return torch.cat([Xa, Zb], dim=1)

    def reverse(self, input):
        Za, Zb = torch.chunk(input, 2, dim=1)
        Xb =  (Zb - self.T(Za)) / self.S(Za)

        return torch.cat([Za, Xb], dim=1)


class INN(nn.Module):

    def __init__(self,  in_features, out_features, n_blocks=1, device='cpu'):   
        assert in_features >= out_features     
        super(INN, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.latent_features = self.in_features - self.out_features

        self.device = device
        
        self.blocks = []

        for _ in range(n_blocks):
            self.blocks.append(Permute(self.in_features, device=self.device))
            self.blocks.append(AffineCoupling(self.in_features, device=self.device))
        
        self.blocks.append(Permute(self.in_features, device=self.device))


    def forward(self, X):
        for block in self.blocks:
            X = block.forward(X)

        return X[:, :self.in_features], X[:, self.in_features:]


    def reverse(self, YZ):
        for block in self.blocks[::-1]:
            YZ = block.reverse(YZ)

        return YZ