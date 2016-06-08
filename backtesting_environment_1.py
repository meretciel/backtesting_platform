
import pandas as pd
import numpy as np
from os import path

import backtesting.analyser as _analyser
import backtesting.portfolio as _portfolio
import backtesting.rebalancer as _rebalancer
import backtesting.simulator as _simulator
from myoperator.basicOperator import *  # TODO: this is not a good practice here, try to find another solution
MyFirstSimulator = _simulator.MyFirstSimulator

import utils.data
import utils.operator


##### Setting ######

# In this section, we need to configure the backtesting environment, such as data directory, stocks of interests,
# variables of interests, etc.


tmp_dir = r'/Users/Ruikun/workspace/backtesting_platform_local/tmp'
data_dir = r'/Users/Ruikun/workspace/backtesting_platform_local/data_2016_06_07'
stocks = ['aapl','msft','mmm','ibm','jpm','wmt','yhoo','gps','ge','f']


# Load variables to the global namespace.

# Manual approach
close    = pd.read_pickle(r'./data/close_dataframe')
adjClose = pd.read_pickle(r'./data/adjClose_dataframe')
rtn      = pd.read_pickle(r'./data/stReturn_dataframe')
open     = pd.read_pickle(r'./data/open_dataframe')
low      = pd.read_pickle(r'./data/low_dataframe')
high     = pd.read_pickle(r'./data/high_dataframe')
volume   = pd.read_pickle(r'./data/volume_dataframe')

# Another approach
utils.data.load_variables(stocks=stocks, data_dir=data_dir, variables=['close','adjClose','open', 'rtn'], global_dict=globals())



#### Create strategies #####
strategyMat = -op_mean(rtn, 5)


#### Prepare the simulation environment ####
portfolio = _portfolio.DollarNeutralPortfolio()
rebalancer = _rebalancer.DollarNeutralPortfolioPeriodicRebalancer(initial_capital=portfolio.initial_capital, frequency=7)
rebalancer.transaction_cost_model.fraction = 0.001



analyser  = _analyser.DollarNeutralPortfolioAnalyser()
priceMat = adjClose.copy()


simulator = MyFirstSimulator(
    priceDF=priceMat,
    portfolio=portfolio,
    rebalancer=rebalancer,
    analyser=analyser
)


simulator.simulate(strategyMat, expression='-op_mean(rtn,5)')
simulator.analyze()
simulator.summarize()


simulator.generate_report(output_path=path.join(tmp_dir, 'test_report.pdf'))

# print (simulator.summarize())

# from datetime import datetime
# prefix = 'test_plot_'
# suffix = datetime.now().strftime('%Y_%m_%d_%H_%M')
# filename = ''.join([prefix, suffix, '.png'])
#
# simulator.plot(save_plot=True, output_path=path.join(tmp_dir, filename))

# simulator.plot()

#
# from datetime import datetime
# prefix = 'test_'
# suffix = datetime.now().strftime('%Y_%m_%d_%H_%M')
# filename = ''.join([prefix, suffix, '.csv'])
#
# simulator.statistics.to_csv(path.join(r'/Users/Ruikun/workspace/backtesting_platform_local/tmp', filename))

# simulator.summarize()


