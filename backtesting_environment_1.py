
import pandas as pd
import numpy as np
from os import path

import backtesting.analyser as _analyser
import backtesting.portfolio as _portfolio
import backtesting.rebalancer as _rebalancer
import backtesting.simulator as _simulator
from myoperator.basicOperator import *  # TODO: this is not a good practice here, try to find another solution
MyFirstSimulator = _simulator.MyFirstSimulator




#### Prepare the data ####

close    = pd.read_pickle(r'./data/close_dataframe')
adjClose = pd.read_pickle(r'./data/adjClose_dataframe')
rtn = pd.read_pickle(r'./data/stReturn_dataframe')
open     = pd.read_pickle(r'./data/open_dataframe')
low      = pd.read_pickle(r'./data/low_dataframe')
high     = pd.read_pickle(r'./data/high_dataframe')
volume   = pd.read_pickle(r'./data/volume_dataframe')




#### Create strategies #####


strategyMat = -op_mean(rtn, 5)


#### Prepare the simulation environment ####
portfolio = _portfolio.DollarNeutralPortfolio()
rebalancer = _rebalancer.DollarNeutralPortfolioPeriodicRebalancer(initial_capital=portfolio.initial_capital, frequency=7)
rebalancer.transaction_cost_model.fraction = 0.001



analyser  = _analyser.DollarNeutralPortfolioAnalyser()

priceMat = adjClose.copy()


simulator = MyFirstSimulator(
    strategyMat=strategyMat,
    priceMat=priceMat,
    portfolio=portfolio,
    rebalancer=rebalancer,
    analyser=analyser
)


simulator.simulate()
simulator.analyze()

simulator.summarize()

# print (simulator.summarize())




# from datetime import datetime
# prefix = 'test_'
# suffix = datetime.now().strftime('%Y_%m_%d_%H_%M')
# filename = ''.join([prefix, suffix, '.csv'])
#
# simulator.statistics.to_csv(path.join(r'/Users/Ruikun/workspace/backtesting_platform_local/tmp', filename))

# simulator.summarize()


