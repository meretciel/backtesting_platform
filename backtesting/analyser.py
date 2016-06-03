


from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd


class AnalyserBase(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self._portfolio = None
        self._portfolio_statistics = None


    @abstractmethod
    def process(self):
        pass

    @abstractmethod
    def summarize(self):
        """
        This function will summarize the results of the analysis.
        """
        pass

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


    def _analyze(self, n, initial_capital, portfolio_value):
        daily_rtn = np.log(portfolio_value[1:]) - np.log(portfolio_value[:-1])
        annual_return = np.log((np.mean(portfolio_value[-10:]) / initial_capital))  * 252. / float(n) - 1.
        volatility    = np.std(daily_rtn) * np.sqrt(252.)
        sharpe_ratio  =annual_return / volatility
        MDD, winLenMDD = self.calculate_maximum_drawdown(portfolio_value)
        return annual_return, sharpe_ratio, MDD, winLenMDD


    def process(self):
        assert self._portfolio and self._portfolio_statistics, 'portfolio and portfolio statistics are required.'

        initial_capital = self._portfolio.initial_capital
        ndays = self._portfolio_statistics.shape[0]
        portfolio_value = self._portfolio_statistics['portfolio_value'].values
        transaction_cost = self._portfolio_statistics['transaction_cost'].values

        n_in_sample = int(ndays * 0.67)


        # full period analysis

        # full period without transaction cost
        portfolio_value_pre_cost = initial_capital + portfolio_value
        self.full_pre_cost = self._analyze(ndays, initial_capital, portfolio_value_pre_cost)

        # full period with transaction cost
        portfolio_value_with_cost = portfolio_value_pre_cost - transaction_cost
        self.full_with_cost = self._analyze(ndays, initial_capital, portfolio_value_with_cost)

        # out-of-sample analysis
        arr = portfolio_value_pre_cost[n_in_sample-10: n_in_sample]
        out_of_sample_initial_capital = np.abs(arr).sum() / 2.

        out_of_sample_portfolio_value_pre_cost  = portfolio_value_pre_cost[n_in_sample:]
        out_of_sample_portfolio_value_with_cost = portfolio_value_with_cost[n_in_sample:]
        out_of_sample_n                         = len(out_of_sample_portfolio_value_pre_cost)

        self.out_of_sample_pre_cost  = self._analyze(out_of_sample_n, out_of_sample_initial_capital, out_of_sample_portfolio_value_pre_cost)
        self.out_of_sample_with_cost = self._analyze(out_of_sample_n, out_of_sample_initial_capital, out_of_sample_portfolio_value_with_cost)


        arr = [list(self.full_pre_cost),
               list(self.full_with_cost),
               list(self.out_of_sample_pre_cost),
               list(self.out_of_sample_with_cost)]
        self._summary = pd.DataFrame(arr,
                                     columns = ['Rtn','IR','MDD','LMDD'],
                                     index   = ['pre_cost','with_cost','oos_pre_cost','oos_with_cost'])



    def summarize(self):
        return self._summary

