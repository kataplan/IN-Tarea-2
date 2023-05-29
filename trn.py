# Training DL via RMSProp+Pinv

import pandas as pd
import numpy as np
import utility as ut

# Training miniBatch for softmax


def train_sft_batch(x, y, W, v, param):
    batch_size = int(param[2])
   
    num_batch = int(x.shape[1]/batch_size)
    costs = []
    for i in range(num_batch):
        idx = get_Idx_n_Batch(batch_size, i)
        xe = x[:, idx]
        ye = y[:, idx]
        z = np.dot(W, xe)
        a = ut.softmax(z)
        gW, cost = ut.gradW_softmax(xe, ye, a, W)
        v, W = ut.updW_sft_rmsprop(W, v, gW, param[1])
        costs.append(cost)
    return W, v, costs


# Softmax's training via SGD with Momentum
def train_softmax(x, y, param):
    w = ut.iniW(x.shape[0], x.shape[0])
    v = np.zeros(w.shape)
    costs = []

    for iter in range(1, int(param[0]+1)):
        idx = np.random.permutation(x.shape[1])
        xe, ye = x[:, idx], y[:, idx]
        w, v, c = train_sft_batch(xe, ye, w, v, param)
        costs.append(np.mean(c))
        if iter % 10 == 0:
            print(f"Iterar-Softmax:{iter}, {costs[iter-1]}")
    return (w, costs)


# AE's Training with miniBatch

def train_ae_batch(x, y, w1, v1, w2, param):
    batch_size = param[3]
    numBatch = np.int16(np.floor(x.shape[1]/batch_size))
    cost_array = []
    for i in range(numBatch):
        idx = get_Idx_n_Batch(batch_size, i)
        xe = x[:, idx]
        a, z = ut.ae_forward(xe, w1, w2, int(param[1]))
        e = a[2]-a[0]
        cost = np.sum(np.sum(e**2))/(2*e.shape[1])
        w1, w2 = ut.backward_ae(a, z, w1, w2, v1, param)
        cost_array.append(cost)
    return (w1, cost_array)


# gets Index for n-th miniBatch
def get_Idx_n_Batch(batch_size, i):
    return np.arange(i*batch_size, i*batch_size + batch_size, dtype=int)


# AE's Training by use miniBatch RMSprop+Pinv
def train_ae(x, y, param, w1, w2, v1):
    maxIter = int(param[2])
    for i in range(1, maxIter):
        xe = x[:, np.random.permutation(x.shape[1])]
        w1, cost = train_ae_batch(xe, y, w1, v1, w2, param)
        if i % 10 == 0:
            print(f"Iterar-AE: {i},{np.mean(cost)}")
    return w1


# SAE's Training
def train_sae(x, y, param):
    W = []

    for hn in range(5, len(param)):
        w1 = ut.iniW(int(param[hn]), x.shape[0])
        w2 = ut.iniW(x.shape[0], int(param[hn]))
        v1 = np.zeros_like(w1)
        w = train_ae(x, y, param, w1, w2, v1)
        z = np.dot(w1, x)
        x = ut.act_function(z, param[1])
        W.append(w)
    return W, x


# load Data for Training
def load_data_trn():
    xe = np.genfromtxt('dtrain_xe.csv', delimiter=',')
    ye = np.genfromtxt('dtrain_ye.csv', delimiter=',')

    return (xe, ye)


def save_w_cost(W, cost):
    np.savez(w_snn.npz)
    xe = np.genfromtxt("data/train.csv", delimiter=",")


# Beginning
def main():
    p_sae, p_sft = ut.load_config()
    xe, ye = load_data_trn()
    W, xr = train_sae(xe.T, ye.T, p_sae)
    Ws, cost = train_softmax(xr.T, ye.T, p_sft)
    ut.save_w_dl(W, Ws, cost)


if __name__ == '__main__':
    main()
