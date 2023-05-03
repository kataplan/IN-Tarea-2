import pandas as pd
import numpy as np
import utility as ut

# Save Data : training and testing


def save_data():

    return


# Binary Label
def binary_label():

    return


# Load data csv
def load_data_csv():

    return ()


# Configuration of the SAEs
def load_config():
    par = np.genfromtxt('cnf_sae.csv', delimiter=',')

    return (par_sae, par_sft)


# Beginning ...
def main():
    Param = ut.load_cnf()
    Data = load_data_csv()
    save_data()


if __name__ == '__main__':
    main()
