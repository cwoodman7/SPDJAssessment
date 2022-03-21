import os
import numpy as np
import pandas as pd


def load_data(filename, sheet_name):
    """This function loads to the data to a pd DataFrame"""
    return pd.read_excel(os.path.join('data', f'{filename}.xlsx'), sheet_name, engine='openpyxl')
    #return pd.read_excel(os.path.join('data', 'SPdata.xls'))


def target(w, *args):
    """This function defines the objective that is minimised during optimisation."""
    x = (w - args[0]) ** 2 / args[0]
    return np.sum(x)


def f_constraint(w, selection_vector):
    """This is a helper function that is used to enforce strict evaluation by the constraint lambda functions"""
    S = np.diag(selection_vector)
    f = lambda s: 0.5 - np.inner(np.dot(S, s), np.ones(len(s)))
    return f(w)
