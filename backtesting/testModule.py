__author__ = 'ruikun'


print 10
print 11


import backtesting_rebalance as br
import pandas as pd
from numpy.random import random
import matplotlib.pyplot as pyplot

weightsDF=None, closeDF=None, returnDF=None, capital=None, period=None, lookback=None



close = pd.read_pickle(r'../data/close_dataframe')
adjClose = pd.read_pickle((r'../data/adjClose_dataframe'))
stReturn = pd.read_pickle(r'../data/stReturn_dataframe')


stReturn.shape == close.shape


capital=10000.
lookback = 5

weights_df_pre = pd.DataFrame(random(close.shape), index=close.index, columns=close.columns)
weights_df = weights_df_pre - weights_df_pre.mean(axis=1)



weights_df.shape == stReturn.shape



reload(br)
res = br.BacktestingRebalanceDollarAmountLevel0(weightsDF=weights_df,closeDF=adjClose, returnDF=stReturn, capital=capital,period=1200, lookback=lookback)
res.plot()




print ('{0}:  {1:.4f}'.format('ir0'.ljust(6), 0.1543453))


for x in range(1,11):
   print '{0:2d} {1:3d} {2:4d}'.format(x, x*x, x*x*x)

