# Training DL via RMSProp+Pinv

import pandas as pd
import numpy as np
import utility as ut


# Training miniBatch for softmax
def train_sft_batch(x, y, W, V, param):
    costo = []
    # for i in range(numBatch):

    return (W, V, costo)


# Softmax's training via SGD with Momentum
def train_softmax(x, y, par1, par2):
    W = ut.iniW(y.shape[0], x.shape[0])
    V = np.zeros(W.shape)

    for Iter in range(1, par1[0]):
        idx = np.random.permutation(x.shape[1])
        xe, ye = x[:, idx], y[:, idx]
        W, V, c = train_sft_batch(xe, ye, W, V, param)

    return (W, Costo)


# AE's Training with miniBatch

def train_ae_batch(x, y, w1, v1, w2, param):
    batch_size = param[3]
    numBatch = np.int16(np.floor(x.shape[1]/batch_size))
    cost = []
    mu = param[4]
    act_f = param[1]
    for i in range(numBatch):
        idx = get_Idx_n_Batch(batch_size, i)
        xe = x[:, idx]
        ye = y[:, idx]
        a, z = ut.ae_forward(xe, w1, w2, act_f)
        gw_1, cost = ut.gradW1(a, z, w1, w2, act_f)
        w1, v1 = ut.updW1_rmsprop(w1, v1, gw_1, mu)
    return (w1, v1, cost)


# gets Index for n-th miniBatch
def get_Idx_n_Batch(batch_size, i):
    return np.arange(i*batch_size, i*batch_size + batch_size, dtype=int)


# AE's Training by use miniBatch RMSprop+Pinv
def train_ae(x, y, param, w1, w2, v1):
    maxIter = int(param[2])
    for i in range(1, maxIter):
        xe = x[:, np.random.permutation(x.shape[1])]
        w1, v1, cost = train_ae_batch(xe, y, w1, v1, w2, param)
        if i % 10 == 0:
            print(f"Iterar-AE: {i},{np.mean(cost)}")
    return (w1)


# SAE's Training
def train_sae(x, y, param):
    W = []

    for hn in range(5, len(param)):
        w1 = ut.iniW(int(param[hn]), x.shape[0])
        w2 = ut.iniW(x.shape[0], int(param[hn]))
        v1 = np.zeros_like(w1)
        w1 = train_ae(x, y, param, w1, w2, v1)
        z = np.dot(w1, x)
        x = ut.act_function(z, param[1])
        W.append(w1)
    return (W, x)


# load Data for Training
def load_data_trn():
    xe = np.genfromtxt('dtrain_xe.csv', delimiter=',')
    ye = np.genfromtxt('dtrain_ye.csv', delimiter=',')

    return (xe, ye)


# Beginning
def main():
    p_sae, p_sft = ut.load_config()
    xe, ye = load_data_trn()
    W, Xr = train_sae(xe.T, ye.T, p_sae)
    Ws, cost = train_softmax(Xr, ye, p_sft, p_sae)
    ut.save_w_dl(W, Ws, cost)


if __name__ == '__main__':
    main()
