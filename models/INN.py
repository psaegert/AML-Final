import torch
import torch.nn as nn
import torch.nn.functional as F

class SkipConnectionNet(nn.Module):
    def __init__(self, in_features, hidden_layer_sizes, out_features):
        super(SkipConnectionNet, self).__init__()
        layers = []

        n_in = in_features
        for n_out in hidden_layer_sizes:
            layers.append(nn.Linear(n_in, n_out))
            layers.append(nn.ReLU())
            n_in = n_out

        layers.append(nn.Linear(n_in, out_features))

        self.model = nn.Sequential(*layers)

    def forward(self, X):
        return self.model(X)


class TwoWayAffineCoupling(nn.Module):
    def __init__(self, features, device=None, dtype=None):
        super(TwoWayAffineCoupling, self).__init__()

        self.features = features

        self.split_size_A = features // 2
        self.split_size_B = self.features - self.split_size_A

        self.logSa = SkipConnectionNet(self.split_size_A, [32, 32], self.split_size_B)
        self.Ta = SkipConnectionNet(self.split_size_A, [32, 32], self.split_size_B)

        self.logSb = SkipConnectionNet(self.split_size_B, [32, 32], self.split_size_A)
        self.Tb = SkipConnectionNet(self.split_size_B, [32, 32], self.split_size_A)


    def forward(self, input):
        '''
        Za = Xa * Sb(Xb) + Tb(Xb)
        Zb = Xb * Sa(Xa) + Ta(Xa)
        '''
        Xa, Xb = torch.chunk(input, 2, dim=1)
        
        log_Sa = self.logSa(Xa)
        log_Sb = self.logSb(Xb)
        Za = Xa * torch.exp(log_Sb) + self.Tb(Xb)
        Zb = Xb * torch.exp(log_Sa) + self.Ta(Xa)

        self.logdet = log_Sa.sum(-1) + log_Sb.sum(-1)

        return torch.cat([Za, Zb], dim=1)

    def inverse(self, input):
        '''
        Xb = (Zb - Ta(Za)) / Sa(Za)
        Xa = (Za - Tb(Zb)) / Sb(Zb)
        '''
        Za, Zb = torch.chunk(input, 2, dim=1)

        Xb = (Zb - self.Ta(Za)) * torch.exp(- self.logSa(Za))
        Xa = (Za - self.Tb(Zb)) * torch.exp(- self.logSb(Zb))

        return torch.cat([Xa, Xb], dim=1)


class RandomPermute(nn.Module):
    def __init__(self, D, device=None, dtype=None):
        super(RandomPermute, self).__init__()

        # number of features
        self.D = D

        self.Q = torch.zeros((self.D, self.D))
        self.Q[torch.randperm(self.D), torch.randperm(self.D)] = torch.ones(self.D)

    def forward(self, X):
        return X.matmul(self.Q)

    def inverse(self, Z):
        return Z.matmul(self.Q.t())


class INN(nn.Module):
    def __init__(self, in_features, out_features, n_blocks=1, device='cpu'):   
        assert in_features >= out_features     
        super(INN, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.latent_features = self.in_features - self.out_features

        self.device = device
        
        self.layers = nn.ModuleList()

        for _ in range(n_blocks):
            self.layers.append(RandomPermute(self.in_features, device=self.device))
            self.layers.append(TwoWayAffineCoupling(self.in_features, device=self.device))
        
        self.layers.append(RandomPermute(self.in_features, device=self.device))


    def forward(self, X):
        self.logdet_sum = 0

        for layer in self.layers:
            X = layer.forward(X)
            if hasattr(layer, 'logdet'):
                self.logdet_sum += layer.logdet

        return X[:, :self.out_features], X[:, self.out_features:]


    def inverse(self, Y, Z):
        YZ = torch.cat([Y, Z], dim=1)
        for layer in self.layers[::-1]:
            YZ = layer.inverse(YZ)

        return YZ