#!/usr/bin/python


import numpy as np
import pandas as pd


def op_lag(df, lag=1) :
    """
    :param df: dataframe representing the economic variables
    :param lag: lag period
    :return: a lagged dataframe

    The lagged dataframe will contain NaN.
    """

    return df.shift(lag)





def op_neutralize(df) :
    """
    :param df: dataframe
    :return: neutralized dataframe such as the sum of each row equals to zero
    """
    row_means = df.mean(axis=1)
    return df.sub(row_means, axis=0)



# rolling mean and std

# This part can be optimized in Cython. It is better to implement the Welford's method
# At this moment, we use the np.convolve function to avoid the for loop when compute the
# simple moving average

def op_mean(df, period) :
    """
    :param df: dataframe
    :param period: the size of the moving window
    :return: new dataframe that contains the simple moving average

    This function compute the time sereis of the moving average for each column
    in the original dataframe.
    """
    nrow, ncol = df.shape
    tmp_arr = np.empty(nrow)
    output_df = pd.DataFrame(np.empty_like(df), index=df.index, columns=df.columns)
    v = np.zeros(period) + 1. / period
    for i_col in xrange(ncol):
        tmp_arr[:] = np.convolve(df.values[:,i_col], v)[:nrow]
        output_df.values[:,i_col] = tmp_arr
    return output_df



def op_rank(df) :
    mat = df.values
    df_new = pd.DataFrame(mat.argsort(axis=1), index=df.index, columns=df.columns)
    return df_new


def op_scale(df):
    row_mean = df.mean(axis=1)
    row_std  = df.std(axis=1)
    df_new = (df.sub(row_mean, axis=0)).div(row_std,axis=0)
    return df_new

