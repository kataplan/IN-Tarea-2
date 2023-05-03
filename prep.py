import pandas as pd
import numpy as np

# Save Data : training and testing


def save_data(data):
    np.savetxt("dtrain_xe.csv", data['xe'], delimiter=",", fmt='%f')
    np.savetxt("dtrain_ye.csv", data['ye'], delimiter=",", fmt='%d')
    np.savetxt("dtest_xv.csv", data['xv'], delimiter=",", fmt='%f')
    np.savetxt("dtest_yv.csv", data['yv'], delimiter="," , fmt='%d')

    return


# Binary Label
def binary_label(y):
    y = y.astype(int)
    max_value = np.max(y)
    binary_array = np.eye(max_value, dtype=int)
    return binary_array[y - 1]


# Load data csv
def load_data_csv(Param):
    d_train = np.genfromtxt('train.csv', delimiter=',')
    d_test = np.genfromtxt('test.csv', delimiter=',')
    data = dict()
    data['xe'] = d_train[:, :-1]
    print(data['xe'].shape)
    data['ye'] = binary_label(d_train[:, -1])
    data['xv'] = d_test[:, :-1]
    data['yv'] = binary_label(d_test[:, -1])

    return data


# Configuration of the SAEs
def load_config():
    par_sae = np.genfromtxt('cnf_sae.csv', delimiter=',')
    par_sft = np.genfromtxt('cnf_sae.csv', delimiter=',')

    return (par_sae, par_sft)


# Beginning ...
def main():
    Param = load_config()
    Data = load_data_csv(Param)
    save_data(Data)


if __name__ == '__main__':
    main()
