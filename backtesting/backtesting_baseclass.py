
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


