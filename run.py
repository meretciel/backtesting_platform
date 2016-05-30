__author__ = 'ruikun'



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
ret      = stReturn


from myoperator.basicOperator import *
from backtesting.backtesting_rebalance import BacktestingRebalanceDollarAmountLevel0 as bktestor
from backtesting.backtesting_rebalance import BacktestingRebalanceLevel1 as mytestor



strategy = op_neutralize(stReturn * 0.2 + 0.8 * open / close)


strategy = op_marketneutralize(-op_mean(ret,5))
expression = 'op_marketneutralize(-op_mean(ret,5))'



res = mytestor(weightsDF=strategy,
               closeDF=adjClose,
               returnDF=stReturn,
               capital=3000,
               period= 3,
               lookback=10,
               expression=expression)
#res = mytestor(weightsDF=strategy, closeDF=adjClose, returnDF=stReturn, capital=3000, period= 3, lookback=10, expression=None)

res.plot()



