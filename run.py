__author__ = 'ruikun'


#test
import numpy as np
import pandas as pd
import os

#




# load data
close    = pd.read_pickle(r'./data/close_dataframe')
adjClose = pd.read_pickle(r'./data/adjClose_dataframe')
stReturn = pd.read_pickle(r'./data/stReturn_dataframe')
open     = pd.read_pickle(r'./data/open_dataframe')
low      = pd.read_pickle(r'./data/low_dataframe')
high     = pd.read_pickle(r'./data/high_dataframe')
volume   = pd.read_pickle(r'./data/volume_dataframe')


from backtesting.backtesting_rebalance import BacktestingRebalanceDollarAmountLevel0 as bktestor


strategy = stReturn * 0.2 + 0.8 * open / close


res = bktestor(weightsDF=strategy, closeDF=adjClose, returnDF=stReturn, capital=3000, period= 60, lookback=5)
res.plot()




