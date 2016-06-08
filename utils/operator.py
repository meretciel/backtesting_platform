
"""
Script: utils/operator.py

In this script, we will implement some utility functions that deal with operators.

"""

import functools
from functools import wraps
import numpy as np
import pandas as pd



def rolling(f):
    @wraps(f)
    def myDecorator(df, win, *args, **kwargs):
        nrow, ncol = df.shape
        output = np.full(df.shape, np.nan)
        for t in xrange(win, nrow):
            output[t, :] = f(df.iloc[t - win:t], *args, **kwargs).values
        return pd.DataFrame(output, columns=df.columns, index=df.index)
    return myDecorator

