
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


    # def plot(self):
    #
    #     plt.plot(pd.to_datetime(self.index), self.arr_valOfPortfolio, color='red')
    #     plt.ylabel('val of portfolio')
    #     plt.xlabel('date')
    #     plt.suptitle('Statistics of Portfolio', fontsize=14, fontweight='bold')
    #
    #     # set ylim
    #     y_min, y_max = (0.94 * np.min(self.arr_valOfPortfolio) , 1.06 * np.max(self.arr_valOfPortfolio))
    #     plt.ylim((y_min, y_max))
    #
    #
    #     # add vertical line
    #     xx = int(0.66 * len(self.index))
    #     plt.plot([self.index[xx],self.index[xx]], [y_min, y_max], color='black')
    #
    #     # add text
    #     delta = (y_max - y_min) / 20.
    #     #xx_topleft = self.index[int(0.73 * len(self.index))]
    #     xx_topleft = self.index[len(self.index) - 200]
    #     yy_topleft = y_min + 7 * delta
    #
    #     plt.text(xx_topleft,yy_topleft,             '{0}:  {1:+.4f}'.format('ir0'.ljust(6),  self.sharpeRatio), fontsize=10)
    #     plt.text(xx_topleft,yy_topleft - delta,     '{0}:  {1:+.4f}'.format('ret0'.ljust(6), self.returnOfPortfolio), fontsize=10)
    #     plt.text(xx_topleft,yy_topleft - 2 * delta, '{0}:  {1:+.4f}'.format('dd0'.ljust(6),  self.maxDrawDown), fontsize=10)
    #     plt.text(xx_topleft,yy_topleft - 3 * delta, '{0}:  {1:+.4f}'.format('ir2'.ljust(6),  self.sharpeRatio_part2), fontsize=10)
    #     plt.text(xx_topleft,yy_topleft - 4 * delta, '{0}:  {1:+.4f}'.format('ret2'.ljust(6), self.returnOfPortfolio_part2), fontsize=10)
    #     plt.text(xx_topleft,yy_topleft - 5 * delta, '{0}:  {1:+.4f}'.format('dd2'.ljust(6),  self.maxDrawDown_part2), fontsize=10)
    #
    #     if self.expression is not None :
    #         plt.set_title(self.expression)
    #
    #
    #     plt.show()
    #     return None
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

        # # add text
        # delta = (y_max - y_min) / 20.
        # #xx_topleft = self.index[int(0.73 * len(self.index))]
        # xx_topleft = self.index[len(self.index) - 200]
        # yy_topleft = y_min + 7 * delta
        #
        # ax.text(xx_topleft,yy_topleft,             '{0}:  {1:+.4f}'.format('ir0'.ljust(6),  self.sharpeRatio), fontsize=10)
        # ax.text(xx_topleft,yy_topleft - delta,     '{0}:  {1:+.4f}'.format('ret0'.ljust(6), self.returnOfPortfolio), fontsize=10)
        # ax.text(xx_topleft,yy_topleft - 2 * delta, '{0}:  {1:+.4f}'.format('dd0'.ljust(6),  self.maxDrawDown), fontsize=10)
        # ax.text(xx_topleft,yy_topleft - 3 * delta, '{0}:  {1:+.4f}'.format('ir2'.ljust(6),  self.sharpeRatio_part2), fontsize=10)
        # ax.text(xx_topleft,yy_topleft - 4 * delta, '{0}:  {1:+.4f}'.format('ret2'.ljust(6), self.returnOfPortfolio_part2), fontsize=10)
        # ax.text(xx_topleft,yy_topleft - 5 * delta, '{0}:  {1:+.4f}'.format('dd2'.ljust(6),  self.maxDrawDown_part2), fontsize=10)

        # if self.expression is not None :
        #     ax.set_title(self.expression)
        #


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

        return None




class BacktestingRebalanceDollarAmountLevel0(BacktestBaseClass):

    def __init__(self, weightsDF=None, closeDF=None, returnDF=None, capital=None, period=None, lookback=None, expression=None):
        self.weightsDF = weightsDF
        self.closeDF  = closeDF
        self.returnDF = returnDF
        self.capital = capital
        self.period=period
        self.lookback=lookback
        self.nrow, self.ncol = self.weightsDF.shape
        self.expression = expression
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

        self.pnlOfPortfolio = finalPnLOfPortfolio
        self.pnlOfPortfolio_part2 = finalPnLOfPortfolio_part2

        self.arr_returnOfPortfolio = retOfPortfolio
        self.arr_valOfPortfolio = valOfPortfolio

        # compute the return of the portfolio


        self.returnOfPortfolio = np.log(1 + finalPnLOfPortfolio / self.capital) * 252. / float(len_period)
        self.returnOfPortfolio_part2 = np.log(1 + finalPnLOfPortfolio_part2 / np.mean(valOfPortfolio[(startIdx-3):(startIdx+3)])) * 252. / float(tmp_nrow)

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





class BacktestingRebalanceLevel1(BacktestBaseClass):

    def __init__(self, weightsDF=None, closeDF=None, returnDF=None, capital=None, period=None, lookback=None,expression=None):
        self.weightsDF = weightsDF
        self.closeDF  = closeDF
        self.returnDF = returnDF
        self.capital = capital
        self.period=period
        self.lookback=lookback
        self.nrow, self.ncol = self.weightsDF.shape
        self.expression=expression
        self.run()
        self.report()





    def run(self):
        weightsMat = self.weightsDF.values[self.lookback:]
        closeMat  = self.closeDF.values[self.lookback:]
        returnMat = self.returnDF.values[self.lookback:]
        tmp_nrow = self.nrow - self.lookback

        self.index = self.weightsDF.index[self.lookback+1 :]


        # compute the value time series of portfolio

        len_period = tmp_nrow - 1

        currentWeights = weightsMat[0,:]
        valOfAssets = self.capital * currentWeights
        arr_valOfPortfolio = np.empty(tmp_nrow - 1)
        arr_pos_valOfPortfolio = np.empty(tmp_nrow-1)
        arr_neg_valOfPortfolio = np.empty(tmp_nrow-1)
        moneyAccount   = np.empty(tmp_nrow-1)
        cash = 0

        for i_row in xrange(1,tmp_nrow):
            #updatedValOfAssets[:] = (1. + returnMat[i_row,:]) * initValOfAssets
            # update the value of each asset
            valOfAssets *= 1. + returnMat[i_row,:]
            pos_valOfPortfolio = np.sum(valOfAssets[currentWeights >= 0])
            neg_valOfPortfolio = np.sum(valOfAssets[currentWeights < 0 ])

            arr_pos_valOfPortfolio[i_row - 1] = pos_valOfPortfolio
            arr_neg_valOfPortfolio[i_row - 1] = neg_valOfPortfolio
            arr_valOfPortfolio[i_row - 1] = pos_valOfPortfolio + neg_valOfPortfolio

            if i_row % self.period == 0 :
                currentValOfPortfolio = arr_valOfPortfolio[i_row-1]

                # update the wegiths
                currentWeights = weightsMat[i_row-1,:]

                if currentValOfPortfolio >= 0 :
                    # positive part has larger value
                    cash += currentValOfPortfolio
                    valOfAssets[:] = (pos_valOfPortfolio - currentValOfPortfolio) * currentWeights
                else:
                    # negative part has larger value
                    cash -= currentValOfPortfolio
                    valOfAssets[:] = pos_valOfPortfolio * currentWeights

                # clear the portfolio value, because now the positive portfolio matches the negative portfolio
                # therefore, the total value of portfolio is zero
                arr_valOfPortfolio[i_row-1] = 0



            moneyAccount[i_row-1] = cash



        self.arr_valOfPortfolio = arr_valOfPortfolio + moneyAccount
        self.arr_pos_valOfPortfolio = arr_pos_valOfPortfolio
        self.arr_neg_valOfPortfolio = arr_neg_valOfPortfolio
        self.arr_moneyAccount = moneyAccount



        # compute the return of the portfolio
        startIdx = int(0.66 * (tmp_nrow-1))

        self.arr_valOfPortfolio = arr_valOfPortfolio + moneyAccount
        self.arr_valOfPortfolio_part2 = self.arr_valOfPortfolio[startIdx:]

        finalValue = self.arr_valOfPortfolio[-1]
        #self.returnOfPortfolio = np.log(0.5 * finalValue / self.capital) * 252. / float(tmp_nrow) - 1.
        #self.returnOfPortfolio_part2 = np.log(0.5 * finalValue / self.arr_valOfPortfolio_part2[0]) * 252. / float(len(self.arr_valOfPortfolio_part2)) - 1.
        #self.returnOfPortfolio_part2 = np.log(1 + 0.5 * finalValue / np.mean(self.arr_valOfPortfolio_part2[:5])) * 252. / float(len(self.arr_valOfPortfolio_part2))

        self.returnOfPortfolio = np.power(0.5 * finalValue / self.capital, 252. / float(tmp_nrow)) - 1
        self.returnOfPortfolio_part2 = np.power(finalValue / self.arr_valOfPortfolio_part2[0], 252. / len(self.arr_valOfPortfolio_part2)) - 1.

        self.maxDrawDown = bkhp.computeMaxDrawDown(self.arr_valOfPortfolio, self.capital)
        self.maxDrawDown_part2 = bkhp.computeMaxDrawDown(self.arr_valOfPortfolio_part2, self.arr_valOfPortfolio_part2[0])


        self.stdOfPortfolio = bkhp.computeStdOfPriceArray(self.arr_valOfPortfolio) * np.sqrt(252)
        self.stdOfPortfolio_part2 = bkhp.computeStdOfPriceArray(self.arr_valOfPortfolio_part2) * np.sqrt(252)

        self.sharpeRatio = self.returnOfPortfolio / self.stdOfPortfolio
        self.sharpeRatio_part2 = self.returnOfPortfolio_part2 / self.stdOfPortfolio_part2

        return None










