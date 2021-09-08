import torch
from torch._C import Value
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList

class NN(nn.Module):
    def __init__(self, in_features, hidden_layer_sizes, out_features):
        super(NN, self).__init__()
        self.layers = nn.ModuleList()

        n_in = in_features
        for n_out in hidden_layer_sizes:
            self.layers.append(nn.Linear(n_in, n_out))
            self.layers.append(nn.ReLU())
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
    def __init__(self, features, device=None, dtype=None):
        super(TwoWayAffineCoupling, self).__init__()

        self.features = features

        self.split_size_A = features // 2
        self.split_size_B = self.features - self.split_size_A

        self.NNa = NN(self.split_size_A, [32, 32], self.split_size_B)
        self.NNb = NN(self.split_size_B, [32, 32], self.split_size_A)


    def forward(self, input):
        '''
        Za = Xa * Sb(Xb) + Tb(Xb)
        Zb = Xb * Sa(Xa) + Ta(Xa)
        '''
        Xa, Xb = torch.chunk(input, 2, dim=1)
        
        log_Sa, Ta = self.NNa(Xa)
        log_Sb, Tb = self.NNb(Xb)
        Za = Xa * torch.exp(log_Sb) + Tb
        Zb = Xb * torch.exp(log_Sa) + Ta

        self.logdet = log_Sa.sum(-1) + log_Sb.sum(-1)

        if torch.isnan(Za).any() or torch.isnan(Zb).any():
            print(f'{torch.where(torch.isnan(Xa)) = }')
            print(f'{torch.where(torch.isnan(Xb)) = }')
            print(f'{torch.min(Xb)=}')
            print(f'{torch.max(Xb)=}')
            
            print(f'{torch.where(torch.isnan(log_Sa)) = }')
            print(f'{torch.where(torch.isnan(log_Sb)) = }')

            print(f'{torch.where(torch.isnan(self.Ta(Xa))) = }')
            print(f'{torch.where(torch.isnan(self.Tb(Xb))) = }')
            
            print(f'{torch.where(torch.isnan(Za)) = }')
            print(f'{torch.where(torch.isnan(Zb)) = }')
            raise ValueError

        return torch.cat([Za, Zb], dim=1)

    def inverse(self, input):
        '''
        Xb = (Zb - Ta(Za)) / Sa(Za)
        Xa = (Za - Tb(Zb)) / Sb(Zb)
        '''

        if torch.isnan(input).any():
            print(f'{input = }')
            raise ValueError

        Za, Zb = torch.chunk(input, 2, dim=1)

        log_Sa, Ta = self.NNa(Za)
        log_Sb, Tb = self.NNb(Zb)
        Xb = (Zb - Ta) * torch.exp(-log_Sa)
        Xa = (Za - Tb) * torch.exp(-log_Sb)

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
            # self.layers.append(RandomPermute(self.in_features, device=self.device))
            self.layers.append(TwoWayAffineCoupling(self.in_features, device=self.device))
        
        # self.layers.append(RandomPermute(self.in_features, device=self.device))


    def forward(self, X):
        self.logdet_sum = 0

        for layer in self.layers:
            X = layer.forward(X)
            if torch.isnan(X).any():
                print(layer._get_name())
                print(torch.where(torch.isnan(X)))
                raise ValueError
            if hasattr(layer, 'logdet'):
                self.logdet_sum += layer.logdet

        return X[:, :self.out_features], X[:, self.out_features:]


    def inverse(self, Y, Z):
        if torch.isnan(Y).any():
            print(f'{Y = }')
            raise ValueError

        YZ = torch.cat([Y, Z], dim=1)
        for layer in self.layers[::-1]:
            YZ = layer.inverse(YZ)

        return YZ