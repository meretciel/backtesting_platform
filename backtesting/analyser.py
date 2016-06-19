

"""
Script: analyser.py

In this script, we will define the analyser class. The main functionality provided by the analyser class is to measuring
the performance of the portfolio in question. It performs the analysis based on the type of portfolio and the rebalance
process.

In the analyser class, we will also define some utility function to measure the performance.

For the current implementation, we will only focus on the following portfolio statistics:
    (1) Sharpe ratio
    (2) Annual return
    (3) Maximum drawdown
    (4) Window length of the maximum drawdown


In the analysis, we will generate two sets of results: (1) statistics of the full backtesting period and
(2) statistics of the last 1/3 of the period.

The second part is considered as the "out-of-sample test". (Strictly speaking, we do not have the in-sample test.



"""

from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np
import pandas as pd

import backtesting.setting as _setting


class AnalyserBase(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self._portfolio = None
        self._portfolio_statistics = None
        self._rebalancer = None
        self._summary = None

    @abstractmethod
    def process(self):
        pass


    def summarize(self, to_print=True):
        s_analyser   = str(self)
        s_portfolio  = str(self._portfolio)
        s_rebalancer = str(self._rebalancer)
        s_aggregated = '\n'.join([s_analyser, s_portfolio, s_rebalancer])

        out = '\n'.join([s_aggregated, "\nPortfolio Performance\n",repr(self._summary)])

        if to_print:
            print(out)
        return out

    @property
    def summary(self):
        return self._summary


    @property
    def portfolio(self):
        return self._portfolio

    @portfolio.setter
    def portfolio(self, value):
        self._portfolio = value

    @property
    def portfolio_statistics(self):
        return self._portfolio_statistics

    @portfolio_statistics.setter
    def portfolio_statistics(self,value):
        self._portfolio_statistics = value

    @property
    def rebalancer(self):
        return self._rebalancer

    @rebalancer.setter
    def rebalancer(self,value):
        self._rebalancer = value



    @staticmethod
    def calculate_maximum_drawdown(arr):
        """
        Args:
            arr: 1d np.array.

        Returns:
            This function return a tuple (MDD, winLenMDD).
                MDD: this is the maximum drawdown of the array
                winLenMDD: this is the window length of the maximum drawdown
        """
        n = arr.size

        MDD = 0.
        MDD_index = -1
        MDD_min_index = -1
        min_index = -1


        min_val = arr.max() + 1.

        for t in xrange(n-1, -1, -1):

            if arr[t] <= min_val:
                min_val = arr[t]
                min_index = t

            new_MDD = arr[t] - min_val
            if new_MDD >= MDD:
                MDD = new_MDD
                MDD_index = t
                MDD_min_index = min_index

        return MDD, MDD_min_index - MDD_index




class DollarNeutralPortfolioAnalyser(AnalyserBase):
    def __str__(self):
        line = 'DollarNeutralPortfolioAnalyser'
        line_ = '='*(len(line))
        return '\n'.join([line_, line, line_])

    def _analyze(self, n , portfolio_value):
        initial_capital = portfolio_value[:self._rebalancer._frequency].mean()
        daily_rtn       = np.log(portfolio_value[1:]) - np.log(portfolio_value[:-1])
        annual_return   = np.exp(np.log((np.mean(portfolio_value[-self._rebalancer._frequency:]) / initial_capital))  * 252. / float(n)) - 1.
        volatility      = np.std(daily_rtn) * np.sqrt(252.)
        sharpe_ratio    = annual_return / volatility
        MDD, winLenMDD  = self.calculate_maximum_drawdown(portfolio_value)
        return annual_return, sharpe_ratio, volatility, MDD, winLenMDD


    def process(self):
        assert self._portfolio is not None
        assert self._rebalancer is not None

        self._summary = None

        if self._portfolio_statistics is None:
            # if the there is no portfolio statistics, we do nothing
            # The implication is the the self._summary = None
            return

        initial_capital = self._portfolio.initial_capital
        ndays = self._portfolio_statistics.shape[0]

        # extract information from the portfolio_statistics which as pd.DataFrame
        account_value_pre_cost   =  initial_capital + self._portfolio_statistics['account_value_pre_cost'].values
        account_value_after_cost =  initial_capital + self._portfolio_statistics['account_value_after_cost'].values

        # We use the last 1/3 part of the data as the out-of-sample data
        n_in_sample = int(ndays * _setting.IN_SAMPLE_PORTION)

        # full period analysis
        self.full_pre_cost = self._analyze(ndays, account_value_pre_cost)
        self.full_with_cost = self._analyze(ndays, account_value_after_cost)

        # out-of-sample analysis
        out_of_sample_portfolio_value_pre_cost  = account_value_pre_cost[n_in_sample:]
        out_of_sample_portfolio_value_with_cost = account_value_after_cost[n_in_sample:]
        out_of_sample_n                         = len(out_of_sample_portfolio_value_pre_cost)

        self.out_of_sample_pre_cost  = self._analyze(out_of_sample_n, out_of_sample_portfolio_value_pre_cost)
        self.out_of_sample_with_cost = self._analyze(out_of_sample_n, out_of_sample_portfolio_value_with_cost)


        arr = [list(self.full_pre_cost),
               list(self.full_with_cost),
               list(self.out_of_sample_pre_cost),
               list(self.out_of_sample_with_cost)]
        self._summary = pd.DataFrame(arr,
                                     columns = ['RTN','IR','VOL', 'MDD','LMDD'],
                                     index   = ['pre_cost','after_cost','oos_pre_cost','oos_after_cost'])





