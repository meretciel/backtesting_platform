

import backtesting.analyser as _analyser
import backtesting.portfolio as _portfolio
import backtesting.rebalancer as _rebalancer
import backtesting.simulator as _simulator

MyFirstSimulator = _simulator.MyFirstSimulator


portfolio = _portfolio.DollarNeutralPortfolio()
rebalancer = _rebalancer.DollarNeutralPortfolioPeriodicRebalancer()
analyser  = _analyser.SimpleAnalyser()



simulator = MyFirstSimulator(
    strategyMat=None,
    priceMat=None,
    portfolio=portfolio,
    rebalancer=rebalancer,
    analyser=analyser
)




