#!/usr/bin/python


import numpy as np
import pandas as pd
import utils.operator

def op_lag(df, lag=1) :
    """
    :param df: dataframe representing the economic variables
    :param lag: lag period
    :return: a lagged dataframe
    The lagged dataframe will contain NaN.
    """
    lag = max(int(lag), 0)

    return df.shift(lag)


def op_neutralize(df) :
    """
    :param df: dataframe
    :return: neutralized dataframe such as the sum of each row equals to zero
    """
    row_means = df.mean(axis=1)
    return df.sub(row_means, axis=0)


# def op_marketneutralize(df) :
#     mat = op_neutralize(df).values
#     mat_pos = np.where(mat > 0, mat,0)
#     mat_neg = np.where(mat <=0, mat, 0)
#     row_sum_pos = mat_pos.sum(axis=1)
#     row_sum_neg = mat_neg.sum(axis=1)
#     nrow,ncol = df.shape
#     mat_pos_new = mat_pos / row_sum_pos.reshape((nrow,1))
#     mat_neg_new = mat_neg / row_sum_neg.reshape((nrow,1))
#     mat_pos_new[np.isnan(mat_pos_new)] = 0
#     mat_neg_new[np.isnan(mat_neg_new)] = 0
#     return pd.DataFrame(mat_pos_new - mat_neg_new, index=df.index, columns=df.columns)
#




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

    period = max(1, int(period))

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


# def op_strategy_add(df1,df2,coef1=0.5,coef2=0.5) :
#     return coef1 * op_scale(df1) + coef2 * op_scale(df2)
