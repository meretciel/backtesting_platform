


"""
script: simulator.py

In this script we will implement the backtesting simulator. Given a strategy, it will generate the useful statistic
to evaluate the input strategy.

"""


from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



class Simulator(object):
    """
    Here we define the interface of a simulator. It has two components and three methods.
    The two components are
        portfolio:  Portfolio object describing the attributes of the portfolio
        rebalancer: Rebalancer object which is responsible for rebalancing the portfolio
        analyser:   Analyser object which provides the functionality to analyze the results.

    The three methods are
        simulate:     simulate the backtesting process
        analyze:      analyze the backtesting results
        summarize:    generate reports of the analysis
    """
    __metaclass__ = ABCMeta


    @abstractmethod
    def simulate(self, strategyMat):
        pass

    @abstractmethod
    def analyze(self):
        pass

    @abstractmethod
    def summarize(self):
        pass





class MyFirstSimulator(Simulator) :

    def __init__(self, priceDF=None, portfolio=None, rebalancer=None, analyser=None, fitness_measure=None,lookback=20):
        if any(v is None for v in [priceDF, portfolio, rebalancer, analyser]):
            raise ValueError("Some parameters are missing.")

        self._portfolio       = portfolio
        self._rebalancer      = rebalancer
        self._analyser        = analyser
        self._fitness_measure = fitness_measure
        self._statistics      = None


        self._priceDF    = priceDF
        self._lookback    = lookback

        # compute the return matrix for future use
        priceDF_lag_1    = self._priceDF.shift(1)
        self._returnDF   = self._priceDF / priceDF_lag_1 - 1.

        self._priceMat  = self._priceDF.values[self._lookback:, :]
        self._returnMat = self._returnDF.values[self._lookback:, :]
        self._date      = self._priceDF.index.values[self._lookback:]




    def simulate(self, strategyDF):
        assert strategyDF.shape == self._priceDF.shape
        self._strategyDF = strategyDF.apply(self._portfolio.normalize_weights, axis=1)

        weightMat     = self._strategyDF.values[self._lookback:, :]
        priceMat      = self._priceMat
        returnMat     = self._returnMat
        dates         = self._date

        # load the priceMat for the rebalancer
        self._rebalancer.load_priceMat(priceMat)

        curr_portfolio    = self._portfolio.get_initial_portfolio(weightMat[0,:])
        ndays             = weightMat.shape[0]
        portfolio_val     = np.zeros(ndays)
        portfolio_val[0]  = curr_portfolio.sum()

        portfolio_position = np.zeros_like(weightMat)
        portfolio_position[0, :] = curr_portfolio

        for t in xrange(1, ndays):
            # Attention: we are at the end of t-th day
            # Do not use information from the future!!!

            # get the curr portfolio position at the end of the t-th day
            curr_portfolio = curr_portfolio * (1. + returnMat[t, :])
            portfolio_val[t]  = np.sum(curr_portfolio)

            # rebalance the portfolio if necessary
            curr_portfolio = self._rebalancer.rebalance_portfolio(t, curr_portfolio,weightMat[t,:])

            portfolio_position[t,:] = curr_portfolio


        # collect all the statistics

        # extract information from rebalancer. This information is more about the transaction cost
        # the result is a DataFrame which has the following columns:
        #   (1) transaction_cost
        #   (2) cash_account
        trading_cost = self._rebalancer.get_trading_cost()
        trading_cost.index = dates

        # combine all the information togeter
        self._statistics = pd.DataFrame(portfolio_position, index=dates, columns=self._strategyDF.columns.map(lambda x: x + '_position'))

        self._statistics = pd.concat([self._statistics,
                                      self._strategyDF.rename(columns=lambda x: x + '_target_weight').reindex(dates),
                                      self._returnDF.rename(columns=lambda x:  x + '_return').reindex(dates),
                                      trading_cost], axis=1)

        # The following information are required by the analyser.
        self._statistics['portfolio_value'] = portfolio_val
        self._statistics['account_value_pre_cost']   = self._statistics.eval('portfolio_value + cash_account')
        self._statistics['account_value_after_cost'] = self._statistics.eval('account_value_pre_cost - transaction_cost')
        self._statistics.index.name = 'date'




    def analyze(self):
        """
        This function will analyze the performance of the portfolio. It first passes the portfolio, portfolio statistics
        and the rebalancer used in the simulation to the analyser. After that, the analyser will process the information and
        generate a summary. For more details, please refer the analyser class.

        """
        self._analyser.portfolio             = self._portfolio
        self._analyser.portfolio_statistics  = self._statistics
        self._analyser.rebalancer            = self._rebalancer

        self._analyser.process()



    def summarize(self):
        self._analyser.summarize()


    @property
    def statistics(self):
        return getattr(self, '_statistics', None)




    #
    #
    #
    # def plot(self):
    #     fig = plt.figure()
    #     fig.suptitle('Statistics of Portfolio', fontsize=14, fontweight='bold')
    #
    #     ax = fig.add_subplot(111)
    #     ax.set_xlabel('date')
    #     ax.set_ylabel('val of portfolio')
    #
    #     ax.plot(pd.to_datetime(self.index), self.arr_valOfPortfolio, color='red')
    #
    #     # set ylim
    #     y_min, y_max = 0.94 * np.min(self.arr_valOfPortfolio), 1.06 * np.max(self.arr_valOfPortfolio)
    #     plt.ylim(y_min,y_max)
    #
    #     # add vertical line
    #     xx = int(0.66 * len(self.index))
    #     ax.plot([self.index[xx],self.index[xx]], [y_min,y_max], color='black')
    #
    #
    #     fig.subplots_adjust(top=0.86)
    #
    #     #row_format = '{}: {:.4f} ' * 3
    #     row_format = r'{}{}  {:.4f}   {}  {:.2f}%   {}  {:.2f}%'
    #
    #
    #     names = ['Info Ratio:', 'Return:', 'Max DD:']
    #     names_formatted = map(lambda x : x.ljust(10), names)
    #     valList_all = [self.sharpeRatio, 100. * self.returnOfPortfolio, 100. * self.maxDrawDown]
    #     valList_part = [self.sharpeRatio_part2, 100. * self.returnOfPortfolio_part2, 100. * self.maxDrawDown_part2]
    #
    #     lineList_all = [None] * 2 * len(names)
    #     lineList_all[::2] = names_formatted
    #     lineList_all[1::2] = valList_all
    #
    #     lineList_part = [None] * 2 * len(names)
    #     lineList_part[::2] = names_formatted
    #     lineList_part[1::2] = valList_part
    #
    #     str_subtitle_line_1 = row_format.format('[all]'.ljust(8),  *lineList_all)
    #     str_subtitle_line_2 = row_format.format('[part]'.ljust(8), *lineList_part)
    #
    #     str_subtitle_stat = str_subtitle_line_1 + '\n' + str_subtitle_line_2
    #
    #     if self.expression is not None :
    #         str_expr = '\nexpression: ' + self.expression
    #         str_subtitle = str_expr + '\n\n' + str_subtitle_stat
    #     else:
    #         str_subtitle = str_subtitle_stat
    #
    #
    #     ax.set_title(str_subtitle)
    #     plt.show()
