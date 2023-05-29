import pandas as pd
import numpy as np
import utility as ut


def save_measure(cm, Fsc):
    np.savetxt('cmatriz.csv', cm, fmt="%d")
    np.savetxt('fscores.csv', Fsc, fmt="%1.25f")


# load data for testing


def load_data_tst():
    xe = np.genfromtxt('dtrain_xe.csv', delimiter=',')
    ye = np.genfromtxt('dtrain_ye.csv', delimiter=',')
    return (xe.T, ye.T)

# load weight of the DL in numpy format


def load_w_dl():
    ws_ae = np.load('w.npz')
    ws_soft = np.load('ws.npz')
    ws = [ws_ae[i] for i in ws_ae.files]
    ws.extend([ws_soft[i] for i in ws_soft.files])
    return ws


# Feed-forward of the DL
def forward_dl(X, W, param):
    for i in range(len(W)):
        X = np.dot(W[i], X)
        if i == len(W)-1:
            X = ut.softmax(X)
        else:
            X = ut.act_function(X, int(param[1]) )
    return X

# Measure


def metricas(Y, Y_predict):
    cm = confusion_matrix(Y, Y_predict)
    precision = np.nan_to_num(
        cm.diagonal() / cm.sum(axis=0), nan=0.0, posinf=1.0)
    recall = np.nan_to_num(cm.diagonal() / cm.sum(axis=1), nan=0.0, posinf=1.0)
    f_score = np.nan_to_num(
        2 * ((precision * recall) / (precision + recall)), nan=0.0, posinf=1.0)
    return cm, np.append(f_score, np.mean(f_score))

# Confusion matrix


def confusion_matrix(Y, Y_predict):
    num_classes = Y.shape[0]
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    max_indices = np.argmax(Y_predict, axis=0)
    Y_pred = np.zeros_like(Y_predict)
    Y_pred[max_indices, np.arange(Y_predict.shape[1])] = 1
    for true_label in range(num_classes):
        for predicted_label in range(num_classes):
            confusion_matrix[true_label, predicted_label] = np.sum(
                (Y[true_label, :] == 1) & (Y_pred[predicted_label, :] == 1))
    return confusion_matrix
# -----------------------------------------------------------------------
# Activation function

# Beginning ...


def main():
    cnf_sae, _ = ut.load_config()
    xv, yv = load_data_tst()
    W = load_w_dl()
    zv = forward_dl(xv, W, cnf_sae)
    cm, Fsc = metricas(yv, zv)
    save_measure(cm, Fsc)


if __name__ == '__main__':
    main()
