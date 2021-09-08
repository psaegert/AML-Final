import torch
import torch.nn as nn
from torch.nn.modules.container import ModuleList


class CouplingNetwork(nn.Module):
    def __init__(self, in_features, hidden_layer_sizes, out_features, device=None):
        super(CouplingNetwork, self).__init__()

        self.layers = nn.ModuleList()
        self.device = device

        n_in = in_features
        for n_out in hidden_layer_sizes:
            self.layers.append(nn.Linear(n_in, n_out))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(0.5))
            n_in = n_out

        self.logS_layers = ModuleList()
        self.logS_layers.append(nn.Linear(hidden_layer_sizes[-1], out_features))
        self.logS_layers.append(nn.Tanh())

        self.T_layers = ModuleList()
        self.T_layers.append(nn.Linear(hidden_layer_sizes[-1], out_features))

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        
        logS = self.logS_layers[0](X)
        logS = self.logS_layers[1](logS)
        
        T = self.T_layers[0](X)

        return logS, T


class TwoWayAffineCoupling(nn.Module):
    def __init__(self, features, coupling_network_layers=None, device=None, dtype=None):
        super(TwoWayAffineCoupling, self).__init__()

        if not coupling_network_layers:
            coupling_network_layers = [64, 64, 64]

        self.features = features

        self.split_size_A = features // 2
        self.split_size_B = self.features - self.split_size_A

        self.CNa = CouplingNetwork(self.split_size_A, coupling_network_layers, self.split_size_B).to(device)
        self.CNb = CouplingNetwork(self.split_size_B, coupling_network_layers, self.split_size_A).to(device)

    def forward(self, input):
        '''
        Za = Xa * Sb(Xb) + Tb(Xb)
        Zb = Xb * Sa(Xa) + Ta(Xa)
        '''
        Xa, Xb = torch.chunk(input, 2, dim=1)
        
        log_Sa, Ta = self.CNa(Xa)
        log_Sb, Tb = self.CNb(Xb)
        Za = Xa * torch.exp(log_Sb) + Tb
        Zb = Xb * torch.exp(log_Sa) + Ta

        self.logdet = log_Sa.sum(-1) + log_Sb.sum(-1)

        return torch.cat([Za, Zb], dim=1)

    def inverse(self, input):
        '''
        Xb = (Zb - Ta(Za)) / Sa(Za)
        Xa = (Za - Tb(Zb)) / Sb(Zb)
        '''
        Za, Zb = torch.chunk(input, 2, dim=1)

        log_Sa, Ta = self.CNa(Za)
        log_Sb, Tb = self.CNb(Zb)
        Xb = (Zb - Ta) * torch.exp(-log_Sa)
        Xa = (Za - Tb) * torch.exp(-log_Sb)

        return torch.cat([Xa, Xb], dim=1)


class RandomPermute(nn.Module):
    def __init__(self, D, device=None, dtype=None):
        super(RandomPermute, self).__init__()

        self.D = D

        self.Q = torch.zeros((self.D, self.D))
        self.Q[torch.randperm(self.D), torch.randperm(self.D)] = torch.ones(self.D)
        self.Q = self.Q.to(device)

    def forward(self, X):
        return X.matmul(self.Q)

    def inverse(self, Z):
        return Z.matmul(self.Q.t())


class INN(nn.Module):
    def __init__(self, in_features, out_features, n_blocks=1, coupling_network_layers=None, device=None):   
        assert in_features >= out_features     
        super(INN, self).__init__()

        self.device = device

        self.in_features = in_features
        self.out_features = out_features
        self.latent_features = self.in_features - self.out_features
        self.layers = nn.ModuleList()

        for _ in range(n_blocks):
            self.layers.append(RandomPermute(self.in_features, device=self.device))
            self.layers.append(TwoWayAffineCoupling(self.in_features, coupling_network_layers=coupling_network_layers, device=self.device))
        
        self.layers.append(RandomPermute(self.in_features, device=self.device))

    def forward(self, X):
        self.logdet_sum = 0

        for layer in self.layers:
            X = layer.forward(X)
            if hasattr(layer, 'logdet'):
                self.logdet_sum += layer.logdet

        return torch.sigmoid(X[:, :self.out_features]), X[:, self.out_features:]

    def inverse(self, Y, Z):
        YZ = torch.cat([Y, Z], dim=1)
        for layer in self.layers[::-1]:
            YZ = layer.inverse(YZ)

        return YZ