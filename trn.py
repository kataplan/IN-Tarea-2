# SNN's Training :

import pandas as pd
import numpy as np
import utility as ut
import prep as pr


# Training miniBatch for softmax
def train_sft_batch(x, y, W, V, param):

    costo = []
    n_batch = int(param[2])
    numBatch = np.int16(np.floor(x.shape[1]/n_batch))
    for n in range(numBatch):
        idx = get_Idx_n_Batch(n, n_batch)
        xe, ye = x[:, slice(*idx)], y[:, slice(*idx)]
        z = np.dot(W, xe)
        a = ut.softmax(z)
        gW, c = ut.gradW_softmax(xe, ye, a, W)
        W, V = ut.updWV_RMSprop_softmax(W, V, gW, param[1])
        costo.append(c)
    return W, V, costo

# Softmax's training via SGD with Momentum


def train_softmax(X, Y, param):
    W = ut.iniW(Y.shape[0], X.shape[0])
    V = np.zeros(W.shape)
    cost = []
    for i in range(1, int(param[0])+1):
        idx = np.random.permutation(X.shape[1])
        xe, ye = X[:, idx], Y[:, idx]
        W, V, c = train_sft_batch(xe, ye, W, V, param)
        cost.append(np.mean(c))
        if i % 10 == 0:
            print(f"Iterar-AE: {i},{np.mean(cost)}")
    return W, cost


# AE's Training with miniBatch
def train_ae_batch(X, W, v, param):
    numBatch = np.int16(np.floor(X.shape[1]/param[3]))
    cost = []
    W[1] = ut.pinv_ae(X, ut.act_function(np.dot(W[0], X), param[1]), param[0])
    for n in range(numBatch):
        idx = get_Idx_n_Batch(n, int(param[3]))
        xe = X[:, slice(*idx)]
        a = ut.forward_ae(xe, W, param)
        gW, Cost = ut.gradW_ae(a, W, param)
        W[0], v[0] = ut.updWV_RMSprop(W, v, gW, param[4])
        cost.append(Cost)
    return W, v, cost


# gets Index for n-th miniBatch
def get_Idx_n_Batch(n, M):
    return n*M, (n*M)+M  # tuple de indices


# AE's Training by use miniBatch RMSprop+Pinv
def train_ae(X, ae_layers, param):
    W, v = ut.iniWs(X.shape[0], ae_layers)
    cost = []
    for i in range(1, int(param[2])+1):
        xe = X[:, np.random.permutation(X.shape[1])]
        W, v, c = train_ae_batch(xe, W, v, param)
        cost.append(np.mean(c))
        if i % 10 == 0:
            print(f"Iterar-Softmax: {i},{np.mean(cost)}")
    return W

# SAE's Training


def train_sae(x, param):
    W = []
    for i, n in enumerate(param[5:]):
        w1 = train_ae(x, int(n), param)
        x = ut.act_function(np.dot(w1[0], x), param[1])
        W.append(w1[0])
    return W, x


# Load data to train the SNN
def load_data_trn():
    xe = np.genfromtxt('dtrain_xe.csv', delimiter=',')
    ye = np.genfromtxt('dtrain_ye.csv', delimiter=',')

    return (xe, ye)


# Beginning ...
def main():
    p_sae, p_sft = ut.load_config()
    xe, ye = load_data_trn()
    W, Xr = train_sae(xe.T, p_sae)
    Ws, cost = train_softmax(Xr, ye.T, p_sft)
    ut.save_w_dl(W, Ws, cost)


if __name__ == '__main__':
    main()
