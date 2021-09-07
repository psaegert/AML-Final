import torch
import torch.nn as nn
import torch.nn.functional as F

class SkipConnectionNet(nn.Module):
    def __init__(self, n_features, layer_sizes):
        super(SkipConnectionNet, self).__init__()
        layers = []

        n_in = n_features
        for n_out in layer_sizes:
            layers.append(nn.Linear(n_in, n_out))
            layers.append(nn.ReLU())
            n_in = n_out

        self.model = nn.Sequential(*layers)

    def forward(self, X):
        return self.model(X)


class AffineCoupling(nn.Module):
    def __init__(self, features, device=None, dtype=None):
        super(AffineCoupling, self).__init__()

        self.features = features

        self.split_size_A = features // 2
        self.split_size_B = self.features - self.split_size_A

        self.S = SkipConnectionNet(self.split_size_A, [32, 32, 32])
        self.T = SkipConnectionNet(self.split_size_A, [32, 32, 32])


    def forward(self, input):
        Xa, Xb = torch.chunk(input, 2, dim=1)
        Zb = self.S(Xa) * Xb + self.T(Xa)

        return torch.cat([Xa, Zb], dim=1)

    def reverse(self, input):
        Za, Zb = torch.chunk(input, 2, dim=1)
        Xb = (Zb - self.T(Za)) / self.S(Za)

        return torch.cat([Za, Xb], dim=1)


class Permute(nn.Module):
    def __init__(self, features, device=None, dtype=None):
        super(Permute, self).__init__()

        self.features = features

        self.Q = torch.zeros((self.features, self.features))
        self.Q[torch.randperm(self.features)] = torch.ones(self.features)

    def forward(self, X):
        return X.matmul(self.Q)

    def reverse(self, Z):
        return Z.matmul(self.Q.t())


class INN(nn.Module):

    def __init__(self, in_features, out_features, n_blocks=1, device='cpu'):   
        assert in_features >= out_features     
        super(INN, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.latent_features = self.in_features - self.out_features

        self.device = device
        
        self.blocks = nn.ModuleList()

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