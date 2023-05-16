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

    act_1 = act_function(np.dot(x,w1.T), act_func)
    act_2 = np.dot(w2, act_1)
    return [x, act_1, act_2.T]


# Calculate Pseudo-inverse
def pinv_ae(x, w1, C):

    return (w2)


# STEP 2: Feed-Backward for AE
def gradW1(a, w1, w2, act_f):

    e = a[2]-a[0]
    Cost = np.sum(np.sum(e**2))/(2*e.shape[1])
    del_2 = np.multiply(e, 1)
    
    gW = np.dot(a[1],del_2)
    return (gW, Cost)


# Update W and V
def updW1_sgdm(w, V, gW, param):

    return


# Sort x y random
def sort_data_ramdom(X, Y):
    XY = np.concatenate((X, Y), axis=1)
    np.random.shuffle(XY)
    X_new, Y_new = np.split(XY, [X.shape[1]], axis=1)
    return X_new, Y_new


# Update AE's weight via RMSprop
def updW1_rmsprop(w, v, gw, mu):
    beta, eps = 0.9, 1e-8
    v_i = beta * v + (1 - beta)* gw**2
    gRMS = np.multiply(gw, 1/(np.sqrt(v_i+eps))).T
    xd =  mu*gRMS    
    w_i = w-xd.T
    return (w_i, v_i)


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
def act_function(z, function_number):
    
    if (function_number == 1):
        h_z = ReLu_function(z).T
    if (function_number == 2):
        h_z = L_ReLu_function(z).T
    if (function_number == 3):
        h_z = ELU_function(z).T
    if (function_number == 4):
        h_z = SELU_function(z).T
    if (function_number == 5):
        h_z = sigmoidal_function(z).T
    return (h_z)


def derivate_act(z, function_number):
    if (function_number == 1):
        h_z = d_ReLu_function(z)
    elif (function_number == 2):
        h_z = d_L_ReLu_function(z)
    elif (function_number == 3):
        h_z = d_ELU_function(z)
    elif (function_number == 4):
        h_z = d_SELU_function(z)
    elif (function_number == 5):
        h_z = d_sigmoidal_function(z)
    return (h_z)


def output_activation(v, h):
    z = np.dot(v, h.T)
    y = 1/(1+np.exp(-z))
    return y.T


def ReLu_function(x):
    return np.where(x > 0, x, 0)


def L_ReLu_function(x):
    return np.where(x < 0, 0.01*x, x)


def ELU_function(x):
    a = 1.6732
    return np.where(x > 0, x, a*(np.exp(x)-1))


def SELU_function(x):
    a = 1.6732
    lam = 1.0507
    return np.where(x > 0, x*lam, a*(np.exp(x)-1))


def sigmoidal_function(z):
    return 1.0/(1.0+np.exp(-z))


def d_ReLu_function(x):
    return np.maximum(0, x)


def d_L_ReLu_function(x):
    return np.where(x < 0, 0.01*x, x)


def d_ELU_function(x):
    a = 1.6732
    return np.where(x > 0, 1, a*np.exp(x))


def d_SELU_function(x):
    lam = 1.0507
    a = 1.6732
    return np.where(x > 0, 1, a*np.exp(x))*lam


def d_sigmoidal_function(z):
    return (np.multiply(1/(1+np.exp(-z)), 1-(1/(1+np.exp(-z))))).T
# -----------------------------------------------------------------------
