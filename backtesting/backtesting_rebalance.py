
#!/usr/bin/python


import numpy as np
import pandas as pd
import backtesting_helperFun as bkhp

import matplotlib.pyplot as plt




class BacktestBaseClass :
    def report(self):
        print('Statistics of the strategy:')
        print ('{0}:  {1:+.4f}'.format('ir0'.ljust(6),  self.sharpeRatio))
        print ('{0}:  {1:+.4f}'.format('ret0'.ljust(6), self.returnOfPortfolio))
        print ('{0}:  {1:+.4f}'.format('dd0'.ljust(6),  self.maxDrawDown))
        print ('{0}:  {1:+.4f}'.format('ir2'.ljust(6),  self.sharpeRatio_part2))
        print ('{0}:  {1:+.4f}'.format('ret2'.ljust(6), self.returnOfPortfolio_part2))
        print ('{0}:  {1:+.4f}'.format('dd2'.ljust(6),  self.maxDrawDown_part2))

    def plot(self):

        plt.plot(pd.to_datetime(self.index), self.arr_valOfPortfolio, color='red')
        plt.ylabel('val of portfolio')
        plt.xlabel('date')
        plt.title('Statistics of Portfolio')

        # set ylim
        y_min, y_max = (0.94 * np.min(self.arr_valOfPortfolio) , 1.06 * np.max(self.arr_valOfPortfolio))
        plt.ylim((y_min, y_max))


        # add vertical line
        xx = int(0.66 * len(self.index))
        plt.plot([self.index[xx],self.index[xx]], [y_min, y_max], color='black')

        # add text
        delta = (y_max - y_min) / 20.
        #xx_topleft = self.index[int(0.73 * len(self.index))]
        xx_topleft = self.index[len(self.index) - 200]
        yy_topleft = y_min + 7 * delta

        plt.text(xx_topleft,yy_topleft,             '{0}:  {1:+.4f}'.format('ir0'.ljust(6),  self.sharpeRatio), fontsize=10)
        plt.text(xx_topleft,yy_topleft - delta,     '{0}:  {1:+.4f}'.format('ret0'.ljust(6), self.returnOfPortfolio), fontsize=10)
        plt.text(xx_topleft,yy_topleft - 2 * delta, '{0}:  {1:+.4f}'.format('dd0'.ljust(6),  self.maxDrawDown), fontsize=10)
        plt.text(xx_topleft,yy_topleft - 3 * delta, '{0}:  {1:+.4f}'.format('ir2'.ljust(6),  self.sharpeRatio_part2), fontsize=10)
        plt.text(xx_topleft,yy_topleft - 4 * delta, '{0}:  {1:+.4f}'.format('ret2'.ljust(6), self.returnOfPortfolio_part2), fontsize=10)
        plt.text(xx_topleft,yy_topleft - 5 * delta, '{0}:  {1:+.4f}'.format('dd2'.ljust(6),  self.maxDrawDown_part2), fontsize=10)


        plt.show()
        return None








class BacktestingRebalanceDollarAmountLevel0(BacktestBaseClass):

    def __init__(self, weightsDF=None, closeDF=None, returnDF=None, capital=None, period=None, lookback=None):
        self.weightsDF = weightsDF
        self.closeDF  = closeDF
        self.returnDF = returnDF
        self.capital = capital
        self.period=period
        self.lookback=lookback
        self.nrow, self.ncol = self.weightsDF.shape
        self.run()
        self.report()
        return None




    def run(self):
        weightsMat = self.weightsDF.values[self.lookback:]
        closeMat  = self.closeDF.values[self.lookback:]
        returnMat = self.returnDF.values[self.lookback:]
        tmp_nrow = self.nrow - self.lookback

        self.index = self.weightsDF.index[self.lookback+1 :]
        # compute the return times series of portfolio
        retOfPortfolio = np.zeros(tmp_nrow - 1)
        for i_row in xrange(tmp_nrow - 1):
            retOfPortfolio[i_row] = np.inner(returnMat[i_row+1], weightsMat[i_row])

        # compute the value time series of portfolio
        valOfPortfolio = np.zeros(tmp_nrow - 1)
        pnlOfPortfolio = 0
        pnlOfPortfolio_part2 = 0
        startIdx = int(0.66 * (tmp_nrow - 1))
        len_period = tmp_nrow - 1

        currentVal = self.capital

        for i_row in xrange(1, tmp_nrow) :
            currentVal = currentVal * (1 + retOfPortfolio[i_row - 1])
            valOfPortfolio[i_row - 1] = currentVal
            if i_row % self.period == 0 :
                pnlOfPortfolio += currentVal - self.capital
                if (i_row >= startIdx) :
                    pnlOfPortfolio_part2 += currentVal - self.capital

                currentVal = self.capital



        finalPnLOfPortfolio = currentVal - self.capital + pnlOfPortfolio
        finalPnLOfPortfolio_part2 = currentVal - self.capital + pnlOfPortfolio_part2

        self.arr_returnOfPortfolio = retOfPortfolio
        self.arr_valOfPortfolio = valOfPortfolio

        # compute the return of the portfolio


        self.returnOfPortfolio = np.log(1 + finalPnLOfPortfolio / self.capital) * 252. / float(len_period)
        self.returnOfPortfolio_part2 = np.log(1 + finalPnLOfPortfolio_part2 / np.mean(valOfPortfolio[(startIdx-3):(startIdx+3)])) * 252. / float(len_period)

        # compute the std
        self.stdOfPortfolio = bkhp.computeStdOfPriceArray(valOfPortfolio) * np.sqrt(252)
        self.stdOfPortfolio_part2 = bkhp.computeStdOfPriceArray(valOfPortfolio[startIdx:]) * np.sqrt(252)

        # compute the Sharpe ratio of the portfolio
        self.sharpeRatio = self.returnOfPortfolio / self.stdOfPortfolio
        self.sharpeRatio_part2 = self.returnOfPortfolio_part2 / self.stdOfPortfolio_part2


        # ompute the maximum drawdown
        self.maxDrawDown = bkhp.computeMaxDrawDown(valOfPortfolio,self.capital)
        self.maxDrawDown_part2  = bkhp.computeMaxDrawDown(valOfPortfolio[startIdx:], valOfPortfolio[startIdx])




        return None






