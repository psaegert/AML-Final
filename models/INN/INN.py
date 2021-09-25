import torch
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList
from tqdm.auto import tqdm

EPSILON = 1e-10


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

        self.logdet = None

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
        Xa, Xb = input[:, :self.split_size_A], input[:, self.split_size_A:]
        
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
        Za, Zb = input[:, :self.split_size_A], input[:, self.split_size_A:]

        log_Sa, Ta = self.CNa(Za)
        log_Sb, Tb = self.CNb(Zb)
        Xb = (Zb - Ta) * torch.exp(-log_Sa)
        Xa = (Za - Tb) * torch.exp(-log_Sb)

        return torch.cat([Xa, Xb], dim=1)


class ConditionalTwoWayAffineCoupling(nn.Module):
    def __init__(self, features, conditions, coupling_network_layers=None, device=None, dtype=None):
        super(ConditionalTwoWayAffineCoupling, self).__init__()

        if not coupling_network_layers:
            coupling_network_layers = [64, 64, 64]

        self.logdet = None

        self.features = features
        self.conditions = conditions

        self.split_size_A = features // 2
        self.split_size_B = self.features - self.split_size_A

        self.CNa = CouplingNetwork(self.split_size_A + self.conditions, coupling_network_layers, self.split_size_B).to(device)
        self.CNb = CouplingNetwork(self.split_size_B + self.conditions, coupling_network_layers, self.split_size_A).to(device)

    def forward(self, input, condition):
        '''
        Za = Xa * Sb(Xb) + Tb(Xb)
        Zb = Xb * Sa(Xa) + Ta(Xa)
        '''
        Xa, Xb = input[:, :self.split_size_A], input[:, self.split_size_A:]
        
        log_Sa, Ta = self.CNa(torch.cat([Xa, condition], dim=1))
        log_Sb, Tb = self.CNb(torch.cat([Xb, condition], dim=1))
        Za = Xa * torch.exp(log_Sb) + Tb
        Zb = Xb * torch.exp(log_Sa) + Ta

        self.logdet = log_Sa.sum(-1) + log_Sb.sum(-1)

        return torch.cat([Za, Zb], dim=1)

    def inverse(self, input, condition):
        '''
        Xb = (Zb - Ta(Za)) / Sa(Za)
        Xa = (Za - Tb(Zb)) / Sb(Zb)
        '''
        Za, Zb = input[:, :self.split_size_A], input[:, self.split_size_A:]

        log_Sa, Ta = self.CNa(torch.cat([Za, condition], dim=1))
        log_Sb, Tb = self.CNb(torch.cat([Zb, condition], dim=1))
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
        # self.latent_features = self.in_features - self.out_features
        self.layers = nn.ModuleList()

        for _ in range(n_blocks):
            self.layers.append(RandomPermute(self.in_features, device=self.device))
            self.layers.append(TwoWayAffineCoupling(self.in_features, coupling_network_layers=coupling_network_layers, device=self.device))
        
        self.layers.append(RandomPermute(self.in_features, device=self.device))

    def forward(self, X):
        self.logdet_sum = torch.zeros((X.size(0))).to(self.device)

        for layer in self.layers:
            X = layer.forward(X)
            if hasattr(layer, 'logdet'):
                self.logdet_sum += layer.logdet

        return torch.sigmoid(X[:, :self.out_features]), X[:, self.out_features:]

    def inverse(self, Y, Z):
        # inverse sigmoid: x = ln(y/(1-y))
        YZ = torch.cat([
            torch.log(torch.clamp(Y, EPSILON, None)/torch.clamp(1-Y, EPSILON, None)),
            Z], dim=1)
        for layer in self.layers[::-1]:
            YZ = layer.inverse(YZ)

        return YZ
    
    def fit(self, X, y, n_epochs=1, batch_size=64, optimizer=None, loss_weights=None, verbose=0, progress_bar_kwargs=None):
        self.train()

        if not optimizer:
            optimizer = Adam(self.parameters(), lr=1e-4)

        if not loss_weights:
            loss_weights = {
                'bce_factor': 1,
                'dvg_factor': 1,
                'logdet_factor': 1,
                'rcst_factor': 1
            }
        
        if not progress_bar_kwargs: 
            progress_bar_kwargs = {}

        loss_history = {
            'weighted_loss': [],
            'bce': [],
            'dvg': [],
            'rcst': [],
            'logdet': []
        }

        # ignores last batch
        n_batches = len(X) // batch_size

        progress_bar_epochs = tqdm(range(n_epochs)) if verbose == 1 else range(n_epochs)
        for i_epoch in progress_bar_epochs:

            progress_bar_batches = tqdm(range(n_batches), desc=f'Epoch {i_epoch}') if verbose == 2 else range(n_batches)
            for i_batch in progress_bar_batches:

                optimizer.zero_grad()
                loss = 0
                
                # --- FORWARD ---
                y_pred, z_pred = self.forward(X[i_batch * batch_size : (i_batch+1) * batch_size])

                bce_loss = F.binary_cross_entropy(y_pred, y[i_batch * batch_size : (i_batch+1) * batch_size]) * loss_weights['bce_factor']
                loss += bce_loss
                loss_history['bce'].append(bce_loss.detach().cpu().numpy())
                
                dvg_loss = torch.mean(torch.sum(z_pred**2, dim=-1)) / 2 * loss_weights['dvg_factor']
                loss += dvg_loss
                loss_history['dvg'].append(dvg_loss.detach().cpu().numpy())
                
                logdet_loss = - torch.mean(self.logdet_sum) * loss_weights['logdet_factor']
                loss += logdet_loss
                loss_history['logdet'].append(logdet_loss.detach().cpu().numpy())
                

                # --- INVERSE ---
                X_pred = self.inverse(y_pred, z_pred)
                
                rcst_loss = F.huber_loss(X_pred, X[i_batch * batch_size : (i_batch+1) * batch_size], delta=2., reduction='mean') * loss_weights['rcst_factor']
                loss += rcst_loss
                loss_history['rcst'].append(rcst_loss.detach().cpu().numpy())


                loss.backward()
                optimizer.step()

                weighted_loss = loss.detach().cpu().numpy()
                loss_history['weighted_loss'].append(weighted_loss)

                if verbose > 0:
                    (progress_bar_epochs if verbose == 1 else progress_bar_batches).set_postfix({
                        'batch': f'{i_batch}/{n_batches}',
                        'weighted_loss': "{}{:.3f}".format("+" if weighted_loss > 0 else "", weighted_loss),
                        'bce': "{}{:.3f}".format("+" if bce_loss > 0 else "", bce_loss / loss_weights['bce_factor']),
                        'dvg': "{}{:.3f}".format("+" if dvg_loss > 0 else "", dvg_loss / loss_weights['dvg_factor']),
                        'rcst': "{}{:.3f}".format("+" if rcst_loss > 0 else "", rcst_loss / loss_weights['rcst_factor']),
                        'logdet': "{}{:.3f}".format("+" if logdet_loss > 0 else "", logdet_loss / loss_weights['logdet_factor']),
                        **progress_bar_kwargs
                    })

        return loss_history

class CINN(nn.Module):
    def __init__(self, n_features, condition_features, n_blocks=1, coupling_network_layers=None, device=None):   
        super(CINN, self).__init__()

        self.device = device

        self.n_features = n_features
        self.condition_features = condition_features
        self.layers = nn.ModuleList()

        for _ in range(n_blocks):
            self.layers.append(RandomPermute(self.n_features, device=self.device))
            self.layers.append(ConditionalTwoWayAffineCoupling(self.n_features, self.condition_features, coupling_network_layers=coupling_network_layers, device=self.device))
        
        self.layers.append(RandomPermute(self.n_features, device=self.device))

    def forward(self, X, Y):
        self.logdet_sum = torch.zeros((X.size(0))).to(self.device)


        for layer in self.layers:
            if hasattr(layer, 'logdet'):
                X = layer.forward(X, Y)
                self.logdet_sum += layer.logdet
            else:
                X = layer.forward(X)


        return X

    def inverse(self, Z, Y):
        for layer in self.layers[::-1]:
            if hasattr(layer, 'logdet'):
                Z = layer.inverse(Z, Y)
            else:
                Z = layer.forward(Z)

        return Z
    
    def fit(self, X, Y, n_epochs=1, batch_size=64, optimizer=None, loss_weights=None, verbose=0, progress_bar_kwargs=None):
        self.train()

        if not optimizer:
            optimizer = Adam(self.parameters(), lr=1e-4)

        if not loss_weights:
            loss_weights = {
                'dvg_factor': 1,
                'logdet_factor': 1,
                'rcst_factor': 1
            }
        
        if not progress_bar_kwargs: 
            progress_bar_kwargs = {}

        loss_history = {
            'weighted_loss': [],
            'dvg': [],
            'rcst': [],
            'logdet': []
        }

        # ignores last batch
        n_batches = len(X) // batch_size

        progress_bar_epochs = tqdm(range(n_epochs)) if verbose == 1 else range(n_epochs)
        for i_epoch in progress_bar_epochs:

            progress_bar_batches = tqdm(range(n_batches), desc=f'Epoch {i_epoch}') if verbose == 2 else range(n_batches)
            for i_batch in progress_bar_batches:

                optimizer.zero_grad()
                loss = 0
                
                # --- FORWARD ---
                z_pred = self.forward(X[i_batch * batch_size : (i_batch+1) * batch_size], Y[i_batch * batch_size : (i_batch+1) * batch_size])
                
                dvg_loss = torch.mean(torch.sum(z_pred**2, dim=-1)) / 2 * loss_weights['dvg_factor']
                loss += dvg_loss
                loss_history['dvg'].append(dvg_loss.detach().cpu().numpy())
                
                logdet_loss = - torch.mean(self.logdet_sum) * loss_weights['logdet_factor']
                loss += logdet_loss
                loss_history['logdet'].append(logdet_loss.detach().cpu().numpy())
                

                # --- INVERSE ---
                X_pred = self.inverse(z_pred, Y[i_batch * batch_size : (i_batch+1) * batch_size])
                
                rcst_loss = F.huber_loss(X_pred, X[i_batch * batch_size : (i_batch+1) * batch_size], delta=2., reduction='mean') * loss_weights['rcst_factor']
                loss += rcst_loss
                loss_history['rcst'].append(rcst_loss.detach().cpu().numpy())


                loss.backward()
                optimizer.step()

                weighted_loss = loss.detach().cpu().numpy()
                loss_history['weighted_loss'].append(weighted_loss)

                if verbose > 0:
                    (progress_bar_epochs if verbose == 1 else progress_bar_batches).set_postfix({
                        'batch': f'{i_batch}/{n_batches}',
                        'weighted_loss': "{}{:.3f}".format("+" if weighted_loss > 0 else "", weighted_loss),
                        'dvg': "{}{:.3f}".format("+" if dvg_loss > 0 else "", dvg_loss / loss_weights['dvg_factor']),
                        'rcst': "{}{:.3f}".format("+" if rcst_loss > 0 else "", rcst_loss / loss_weights['rcst_factor']),
                        'logdet': "{}{:.3f}".format("+" if logdet_loss > 0 else "", logdet_loss / loss_weights['logdet_factor']),
                        **progress_bar_kwargs
                    })

        return loss_history