# My Utility : auxiliars functions

import pandas as pd
import numpy as np


# Initialize one-wieght
def iniW(next, prev):
    r = np.sqrt(6/(next + prev))
    w = np.random.rand(next, prev)
    w = w*2*r-r
    return (w)


# STEP 1: Feed-forward of AE
def ae_forward(x, w1, w2, act_func):
    A = [x]
    Z = [np.dot(w1, A[0])]
    A.append(act_function(Z[0], act_func))
    Z.append(np.dot(w2, A[1]))
    A.append(Z[1])
    return A, Z


# Calculate Pseudo-inverse
def pinv_ae(x, w1, C):

    return (w2)

# STEP 2: Feed-Backward for AE
def gradW1(a, z, w1, w2, act_f):
    e = a[2]-a[0]
    Cost = np.sum(np.sum(e**2))/(2*e.shape[1])
    del_2 = e
    gW = np.dot(w2.T, del_2) * deriva_act(z[0], act_f)
    gW = np.dot(gW, a[0].T)
    return (gW, Cost)


# Update W and V
def updW1_sgdm(w, V, gW, param):

    return


# Sort x y random
def sort_data_ramdom(X, Y):
    XY = np.concatenate((X, Y), axis=0)
    np.random.shuffle(XY)
    X, Y = np.split(XY, [X.shape[0]], axis=0)
    return X, Y


# Update AE's weight via RMSprop
def updW1_rmsprop(w, v, gw, mu):
    beta, eps = 0.9, 1e-8
    v = beta * v + (1 - beta) * gw**2
    gRMS = (1/(np.sqrt(v + eps))) * gw
    w = w - mu*gRMS
    return (w, v)


# Update Softmax's weight via RMSprop
def updW_sft_rmsprop(w, v, gw, mu):
    beta, eps = 0.9, 1e-8

    return (w, v)


# Softmax's gradient
def gradW_softmax(x, y, a):
    ya = y*np.log(a)

    return (gW, Cost)


def load_config():
    par_sae = np.genfromtxt('cnf_sae.csv', delimiter=',')
    par_sft = np.genfromtxt('cnf_sae.csv', delimiter=',')
    return (par_sae, par_sft)


# Calculate Softmax
def softmax(z):
    exp_z = np.exp(z-np.max(z))
    return (exp_z/exp_z.sum(axis=0, keepdims=True))


# save weights SAE and costo of Softmax
def save_w_dl(W, Ws, cost):
    return


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
