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
    numBatch = np.int16(np.floor(x.shape[1]/param[0]))
    cost = []
    mu = param[4]
    act_f = param[1]
    x = x.T
    for i in range(numBatch):
        print("x = ", x.shape)
        print("w1 = ", w1.shape)
        print("w2 = ", w2.shape)
        print("v = ", v1.shape)
        a = ut.ae_forward(x, w1, w2, act_f)
        # i = 0
        # for act in a:
        #     print("a_", i," = ", act.shape)
        #     i = +1

        gw_1,cost = ut.gradW1(a, w1, w2, act_f)
        print("gw1 = ", gw_1.shape)

        w1, v1 = ut.updW1_rmsprop(w1, v1, gw_1, mu)
    return (w1, v1, cost)


# Function to get the indices of the n-th batch
def get_Idx_n_Batch(n, M):
    start = (n-1)*M
    end = n*M
    return np.arange(start, end)


# AE's Training by use miniBatch RMSprop+Pinv
def train_ae(x, y, hn, param):
    nodes_encoded = int(param[hn])
    w1 = ut.iniW(nodes_encoded, x.shape[0])
    w2 = ut.iniW(x.shape[0], nodes_encoded)
    v1 = np.zeros_like(w1)
    maxIter = int(param[2])

    for i in range(1, maxIter):
        x, y = ut.sort_data_ramdom(x, y)
        w1, v1, c = train_ae_batch(x, y, w1, v1, w2, param)

    return (w1.T)


# SAE's Training
def train_sae(x, y, param):
    W = []

    for hn in range(5, len(param)):
        w1 = train_ae(x, y, hn, param)
        x = ut.act_function(w1, x, param)
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
    W, Xr = train_sae(xe, ye, p_sae)
    Ws, cost = train_softmax(Xr, ye, p_sft, p_sae)
    ut.save_w_dl(W, Ws, cost)


if __name__ == '__main__':
    main()
