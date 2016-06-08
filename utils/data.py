

"""
Script: utils/data.py

In this script, we define utility function for data manipulations.

"""

import os
from os import path
import re
import pandas as pd



# Files to skip. We do not necessarily want to all the data in the directory
_file_to_skip = ['fundamental_data_of_stocks.csv']




def load_variables(stocks=None, data_dir=None, variables=None, global_dict=None):
    """
    This function will load the files in the directory and add them to the global namespace. This function will be called
    when initializing the simulation environment. Variables such as stock price are only available after this function
    is called.

    Args:
        stocks:     list. This specifies which stocks we want to work on and the function will extract information of
                    stocks listed in this variable.
        data_dir:   directory. This is the place where the variables files are stored.
        variables:  list. This specifies which variables we want to select. The default value is None which means we
                    select all the variables.
        global_dict:  dictionary. This is the global namespace. The variables will be load into this namespace.


    """
    assert stocks
    assert data_dir
    assert global_dict

    stocks = [x.upper() for x in stocks]
    available_file = [x for x in os.listdir(data_dir) if not x in _file_to_skip]

    # load the DataFrame and add it to the global namespace
    for filename in available_file:
        var_name = re.sub('\.csv$', '', filename)
        # if the variables is not the one use wants to select, we skip the loading process
        if variables and not var_name in variables:
            continue
        # read the files and set index to Date
        df = pd.read_csv(path.join(data_dir, filename))
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index(['Date'])
        # select stocks
        if stocks:
            df = df.reindex(columns=stocks)
        # add to global namespace
        global_dict[var_name] = df.copy()


if __name__ == '__main__':
    load_variables()
    data_dir = r'/Users/Ruikun/workspace/backtesting_platform_local/data_2016_06_07'
    stocks = ['aapl','msft','mmm','ibm','jpm','wmt','yhoo','gps','ge','f']
    load_variables(stocks=stocks, data_dir=data_dir)