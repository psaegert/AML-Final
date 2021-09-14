import os

import numpy as np
import pickle
import time

import INN
import torch
from torch.optim import Adam

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss

import GPy
import optunity as opt
import sobol as sb

import scipy.stats as stats


print('Preparing...')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

print('Loading Train Data...')
with open('../../data/data_train.pt', 'rb') as file:
    X_train, y_train = pickle.load(file)

print(f'{X_train.shape = }')
print(f'{y_train.shape = }')


print('Setting Parameters...')
INN_parameters = {
    'in_features': X_train.shape[1],
    'out_features': y_train.shape[1],
    'device': device
}

loss_weights = {
    'bce_factor': 10,
    'dvg_factor': 1,
    'logdet_factor': 1,
    'rcst_factor': 1
}

lr = 5e-4
n_epochs = 8
batch_size = 1024

hyperparameter_search_space_boundaries = {
    'n_blocks': [1, 12],
    'n_coupling_network_hidden_layers': [1, 5],
    'n_coupling_network_hidden_nodes': [4, 512 + 256],
}


def scale_hyperparameters(hyperparameters):
    return np.array([h * (boundaries[1] - boundaries[0]) + boundaries[0] for h, boundaries in zip(hyperparameters, hyperparameter_search_space_boundaries.values())])


def get_mean_CV_Score(score_function, hyperparameters, progress_bar_kwargs=None):
    n_blocks, n_coupling_network_hidden_layers, n_coupling_network_hidden_nodes = hyperparameters

    kf = KFold(n_splits=5, shuffle=True, random_state=20210927)

    log_loss_list = np.empty(5, dtype=np.float64)

    for split_index, (fit_index, val_index) in enumerate(kf.split(X_train)):
        # create splits
        X_fit, X_val = X_train[fit_index], X_train[val_index]
        y_fit, y_val = torch.Tensor(y_train[fit_index]).to(device), y_train[val_index]

        # scale features
        sc_X_fit = StandardScaler()
        X_fit_scaled = torch.Tensor(sc_X_fit.fit_transform(X_fit)).to(device)
        X_val_scaled = torch.Tensor(sc_X_fit.transform(X_val)).to(device)

        # create classifier
        inn = INN.INN(**INN_parameters, n_blocks=n_blocks, coupling_network_layers=[n_coupling_network_hidden_nodes] * n_coupling_network_hidden_layers).to(device)

        inn.train()

        # fit
        inn.fit(
            X_fit_scaled,
            y_fit,
            n_epochs=n_epochs,
            batch_size=batch_size,
            optimizer=Adam(inn.parameters(), lr=lr), 
            loss_weights=loss_weights,
            verbose=1,
            progress_bar_kwargs=progress_bar_kwargs
        )

        inn.eval()

        del X_fit_scaled, y_fit

        # evaluate
        n_batches = len(X_val) // batch_size
        y_proba_pred = np.empty((0, 2))
        for i_batch in range(n_batches + 1):
            y_proba_pred_new = inn.forward(X_val_scaled[i_batch * batch_size: (i_batch+1) * batch_size])[0].detach().cpu().numpy()
            y_proba_pred = np.concatenate([y_proba_pred, y_proba_pred_new], axis=0)

        # log_loss_list[split_index] = score_function(y_val, y_proba_pred)
        # hosp and death
        log_loss_list[split_index] = score_function(y_val[:, 0], y_proba_pred[:, 0]) + score_function(y_val[:, 1], y_proba_pred[:, 1])

        del inn, X_val_scaled

    return np.mean(log_loss_list)


def expected_improvement(n_blocks, n_coupling_network_hidden_layers, n_coupling_network_hidden_nodes, gp):
    # compute E(q) and Var(q)
    E_pred, Var_pred = gp.predict_noiseless(np.array([[n_blocks, n_coupling_network_hidden_layers, n_coupling_network_hidden_nodes]]))

    # compute gamma with the STD(q)
    γ = (E_best - E_pred) / np.sqrt(Var_pred)

    # return Expected Improvement
    return (np.sqrt(Var_pred) * (γ * stats.norm.cdf(γ) + stats.norm.pdf(γ)))[0]


def initialize_GP(n_samples, progress=0):
    Q_init = np.empty((n_samples, len(hyperparameter_search_space_boundaries)))
    E_init = np.empty((n_samples, 1))

    # initialize with sobol sequence between 0 and 1
    for i in range(n_samples):
        for j, boundaries in enumerate(hyperparameter_search_space_boundaries.values()):
            Q_init[i, j] = sb.i4_sobol(len(hyperparameter_search_space_boundaries), i)[0][j]

    # compute scores for the initial hyperparameters
    for i, hyperparameters in enumerate(Q_init):

        # skip the ones that have already been computed
        if progress > i:
            continue

        # scale hyperparameters according to their bounds and convert them to integers
        hyperparameters_scaled = scale_hyperparameters(hyperparameters).round().astype(int)

        # print the status
        hyperparameters_dict = {key: hyperparameters_scaled[i] for i, key in enumerate(hyperparameter_search_space_boundaries.keys())}
        print(f'{i+1}/{len(Q_init)}: {hyperparameters_dict}')
        time.sleep(0.35)
        
        # compute cv score
        E_init[i, :] = get_mean_CV_Score(log_loss, hyperparameters_scaled)
        print(f'score: {E_init[i, :]}')
        progress += 1

        # save checkpoint
        print('Storing Checkpoint...')
        with open(f'../../hyperparameter_results/INN.pt', 'wb') as file:
            pickle.dump((Q_init, E_init), file)
        with open(f'../../hyperparameter_results/INN_progress.pt', 'wb') as file:
            pickle.dump(progress, file)
        print('Stored Checkpoint...')

    return Q_init, E_init


initial_n_samples = 8
additional_n_samples = 24


# load checkpoint if possible
if os.path.isfile('../../hyperparameter_results/INN.pt') and os.path.isfile('../../hyperparameter_results/INN_progress.pt'):
    print('Loading Checkpoint...')
    with open('../../hyperparameter_results/INN.pt', 'rb') as file:
        Q, E = pickle.load(file)
    with open('../../hyperparameter_results/INN_progress.pt', 'rb') as file:
        progress = pickle.load(file)
    print('Loaded Checkpoint')
else:
    progress = 0

# if not all initial hyperparameters have been tested, continue testing them
if progress < initial_n_samples:
    print(f"Initializing GP...")
    time.sleep(0.3)
    Q, E = initialize_GP(initial_n_samples, progress=progress)
    progress = initial_n_samples

# main GP training loop
print('Training GP...')
for k in range(progress - initial_n_samples, additional_n_samples):
    # train Gaussian Process
    GP = GPy.models.GPRegression(Q, E, kernel=GPy.kern.Matern52(3))
    GP.optimize(messages=False)

    # determine E_best (minimum value of E)
    E_best = np.min(E)

    # determine q_new (q with maximum expected improvement)
    optimizer_output = opt.maximize(
        lambda **kwargs: expected_improvement(gp=GP, **kwargs),
        **{k: [0, 1] for k in hyperparameter_search_space_boundaries.keys()}
    )[0]

    # extract and scale new 'optimal' hyperparameters
    q_new = np.array([optimizer_output[k] for k in hyperparameter_search_space_boundaries.keys()]).ravel()
    q_new_scaled = scale_hyperparameters(q_new).round().astype(int)

    # only for integer values: if the new hyperparameters have already been tested, the algorithm converged
    for q in Q:
        if (q_new == q).all():
            print('GP Converged early.')
            break

    # print status
    hyperparameters_dict = {key: q_new_scaled[i] for i, key in enumerate(hyperparameter_search_space_boundaries.keys())}
    print(f'{k+1}/{additional_n_samples}: {hyperparameters_dict}')
    time.sleep(0.3)

    # add q_new to the training set Q
    Q = np.vstack((Q, q_new))

    # add value to E
    E = np.vstack((E, get_mean_CV_Score(log_loss, q_new_scaled).reshape(-1, 1)))
    print(f'score: {E[-1, :]}')

    # save checkpoint
    progress += 1
    print('Storing Checkpoint...')
    with open(f'../../hyperparameter_results/INN.pt', 'wb') as file:
        pickle.dump((Q, E), file)
    with open(f'../../hyperparameter_results/INN_progress.pt', 'wb') as file:
        pickle.dump(progress, file)
    print('Stored Checkpoint...')
