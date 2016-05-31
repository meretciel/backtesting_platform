


"""
script: simulator.py

In this script we will implement the backtesting simulator. Given a strategy, it will generate the useful statistic
to evaluate the input strategy.

"""




import numpy as np
import pandas as pd
import backtesting_helperFun as bkhp
import matplotlib.pyplot as plt





class SimulatorBase(object) :

    def __init__(self):

        self._rebalance         = None
        self._transaction_cost  = None
        self._market_impact     = None
        self._strategy          = None
        self._initial_capital   = None
        self._lookback          = None
        self._statistic         = None



        self._config           = {

        }

        self._weightDF   = None
        self._adjCloseDF = None
        self._returnDF   = None



    def simulate(self):


        weightMat     = self._weightDF.values[self._lookback:]
        adjCloseMat   = self._adjCloseDF.values[self._lookback:]
        returnMat     = self._returnDF.values[self._lookback:]

        curr_portfolio     = weightMat[0,:] * self._initial_capital
        prev_portfolio     = curr_portfolio.copy()

        periods     = weightMat.shape[0]

        portfolio_value    = np.zeros(periods)
        portfolio_value[0] = self._initial_capital


        # in sample simulation
        for t_day in xrange(1, periods):
            curr_portfolio         = prev_portfolio * (1. + returnMat[t_day,:])
            curr_portfolio_value   = np.sum(curr_portfolio)
            portfolio_value[t_day] = curr_portfolio_value

            # Rebalance the portoflio
            curr_portfolio = self._rebalance.rebalance_portfolio(t_day, weightMat, curr_portfolio, adjCloseMat, returnMat)
            prev_portfolio = curr_portfolio.copy()



        self._portfolio_statistic = pd.DataFrame({
            'date':             self._strategy.index.values[self._lookback:],
            'portfolio_val':    portfolio_value,
            'trading_cost':     self._rebalance.get_trading_cost(),
            'transaction_cost': self._rebalance.get_transaction_cost(),
            'market_impact':    self._rebalance.get_market_impact_estimate,
        })


    def analyze(self):
        """
        This function will analyze the results. The input is self._portfolio_statistic. It is a pd.DataFrame and should
        contain the following columns:
            date:   date
            portfolio_val:    This is the portfolio value in dollar amount
            trading_cost:     This is the total trading cost. Every time we re-balance the portfolio, there will be trading cost.
            transaction_cost: This is the cost to place the trade


        For the current implementation, we will only focus on the portfolio statistics. In particular, we are interested
        in the following measures:
            (1) Sharpe ratio
            (2) Annual return
            (3) Maximum drawdown
            (4) Window length of the maximum drawdown


        In the analysis, we will generate two sets of results: (1) statistics of the full backtesting period and
        (2) statistics of the last 1/3 of the period.

        The second part is considered as the "out-of-sample test". (Strictly speaking, we do not have the in-sample test.)

        """

        df_work = self._portfolio_statistic

        nrow, ncol = df_work.shape








    def report(self):
        print('Statistics of the strategy:')
        print ('{0}:  {1:+.4f}'.format('ir0'.ljust(6),  self.sharpeRatio))
        print ('{0}:  {1:+.4f}'.format('ret0'.ljust(6), self.returnOfPortfolio))
        print ('{0}:  {1:+.4f}'.format('dd0'.ljust(6),  self.maxDrawDown))
        print ('{0}:  {1:+.4f}'.format('ir2'.ljust(6),  self.sharpeRatio_part2))
        print ('{0}:  {1:+.4f}'.format('ret2'.ljust(6), self.returnOfPortfolio_part2))
        print ('{0}:  {1:+.4f}'.format('dd2'.ljust(6),  self.maxDrawDown_part2))



    def plot(self):
        fig = plt.figure()
        fig.suptitle('Statistics of Portfolio', fontsize=14, fontweight='bold')

        ax = fig.add_subplot(111)
        ax.set_xlabel('date')
        ax.set_ylabel('val of portfolio')

        ax.plot(pd.to_datetime(self.index), self.arr_valOfPortfolio, color='red')

        # set ylim
        y_min, y_max = 0.94 * np.min(self.arr_valOfPortfolio), 1.06 * np.max(self.arr_valOfPortfolio)
        plt.ylim(y_min,y_max)

        # add vertical line
        xx = int(0.66 * len(self.index))
        ax.plot([self.index[xx],self.index[xx]], [y_min,y_max], color='black')


        fig.subplots_adjust(top=0.86)

        #row_format = '{}: {:.4f} ' * 3
        row_format = r'{}{}  {:.4f}   {}  {:.2f}%   {}  {:.2f}%'


        names = ['Info Ratio:', 'Return:', 'Max DD:']
        names_formatted = map(lambda x : x.ljust(10), names)
        valList_all = [self.sharpeRatio, 100. * self.returnOfPortfolio, 100. * self.maxDrawDown]
        valList_part = [self.sharpeRatio_part2, 100. * self.returnOfPortfolio_part2, 100. * self.maxDrawDown_part2]

        lineList_all = [None] * 2 * len(names)
        lineList_all[::2] = names_formatted
        lineList_all[1::2] = valList_all

        lineList_part = [None] * 2 * len(names)
        lineList_part[::2] = names_formatted
        lineList_part[1::2] = valList_part

        str_subtitle_line_1 = row_format.format('[all]'.ljust(8),  *lineList_all)
        str_subtitle_line_2 = row_format.format('[part]'.ljust(8), *lineList_part)

        str_subtitle_stat = str_subtitle_line_1 + '\n' + str_subtitle_line_2

        if self.expression is not None :
            str_expr = '\nexpression: ' + self.expression
            str_subtitle = str_expr + '\n\n' + str_subtitle_stat
        else:
            str_subtitle = str_subtitle_stat


        ax.set_title(str_subtitle)
        plt.show()



