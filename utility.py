# My Utility : auxiliars functions

import pandas as pd
import numpy as np
import prep as pr

# load config param


def load_config():
    par_sae = np.genfromtxt('cnf_sae.csv', delimiter=',')
    par_sft = np.genfromtxt('cnf_sae.csv', delimiter=',')
    return (par_sae, par_sft)

# Initialize weights for SNN-SGDM


def iniWs(inshape, layer_node):
    W1 = iniW(layer_node, inshape)
    W2 = iniW(inshape, layer_node)
    W = [W1, W2]
    V = []
    for i in range(len(W)):
        V.append(np.zeros(W[i].shape))

    return W, V


# Initialize weights for one-layer

def iniW(next, prev):
    r = np.sqrt(6/(next + prev))
    w = np.random.rand(next, prev)
    w = w*2*r-r
    return w


# Feed-forward of SNN
def forward_ae(x, W, param):
    act_f = param[1]
    A = []
    z = []
    Act = []
    z.append(x)
    A.append(x)

    for i in range(len(W)):
        x = np.dot(W[i], x)
        z.append(x)
        if i == 0:
            x = act_function(x, act_f)
        A.append(x)
    Act = [A, z]
    return Act


# Activation function
def act_function(Z, act_func):
    if act_func == 1:
        return np.maximum(0, Z)
    if act_func == 2:
        return np.maximum(0.01 * Z, Z)
    if act_func == 3:
        return np.where(Z > 0, Z, 1 * (np.exp(Z) - 1))
    if act_func == 4:
        lam = 1.0507
        alpha = 1.6732
        return np.where(Z > 0, Z, alpha * (np.exp(Z) - 1)) * lam
    if act_func == 5:
        return 1 / (1 + np.exp(-Z))


# Derivatives of the activation function
def deriva_act(A, act_func):
    if act_func == 1:
        return np.where(A >= 0, 1, 0)
    if act_func == 2:
        return np.where(A >= 0, 1, 0.01)
    if act_func == 3:
        return np.where(A >= 0, 1, 0.01 * np.exp(A))
    if act_func == 4:
        lam = 1.0507
        alpha = 1.6732
        return np.where(A > 0, 1, alpha * np.exp(A)) * lam
    if act_func == 5:
        s = act_function(A, act_func)
        return s * (1 - s)


# Calculate Pseudo-inverse
def pinv_ae(x, H, C):
    inv = np.linalg.pinv(np.dot(H, H.T) + (1/C))
    w2 = np.linalg.multi_dot([x, H.T, inv])
    return (w2)


# Feed-Backward of SNN
def gradW_ae(a, W, param):
    act_f = param[1]
    L = len(a[0]) - 1
    M = param[3]
    e = a[0][L] - a[0][0]
    Cost = np.sum(np.square(e)) / (2 * M)
    delta = e
    gW_l = np.dot(delta, a[0][L-1].T) / M
    gW = [gW_l]
    dz1 = np.dot(W[1].T, delta)
    dz2 = deriva_act(a[1][1], act_f)
    dz3 = a[0][0].T
    gW_l = np.dot(np.multiply(dz1, dz2), dz3) / M
    gW.append(gW_l)
    gW.reverse()

    return gW, Cost


# Update W and V
def updWV_RMSprop(W, V, gW, tasa=0.1):
    e = 1e-8
    beta = 0.9
    for i in range(len(W)):
        V[i] = (beta * V[i]) + ((1-beta)*gW[i]**2)
        W[i] = W[i] - ((tasa / np.sqrt(V[i]+e)) * gW[i])
    return W[0], V[0]


def updWV_RMSprop_softmax(w, v, gW, mu):
    e = 1e-8
    beta = 0.9
    v = (beta * v) + ((1-beta)*gW**2)
    w = w - ((mu / np.sqrt(v+e)) * gW)
    return w, v



# Softmax's gradient
def gradW_softmax(x, y, a, W):
    M = y.shape[1]
    Cost = -(np.sum(np.sum(np.multiply(y, np.log(a)), axis=0)/2))/M
    gW = -(np.dot(y-a, x.T))/M 
    return gW, Cost


# Calculate Softmax
def softmax(z):
    exp_z = np.exp(z-np.max(z))
    return (exp_z/np.sum(exp_z, axis=0, keepdims=True))


# Save weights and MSE  of the SNN
def save_w_dl(w, ws, Cost):
    np.savez('w.npz', w[0], w[1])
    np.savez('ws.npz', ws)
    df = pd.DataFrame(Cost)
    df.to_csv('costo.csv', index=False, header=False)

    return

# -----------------------------------------------------------------------
