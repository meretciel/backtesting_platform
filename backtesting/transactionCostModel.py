
"""
Script transactionCostModel.py

This function will implement some of the basic transaction cost model. It is one of the attributes of the Portfolio
class.


"""



from abc import ABCMeta, abstractmethod
import numpy as np

class TransactionCostModel(object):
    __metaclass__ = ABCMeta

    def __get__(self, instance, owner):
        return self

    @abstractmethod
    def estimate_cost(self, curr_t, priceMat, curr_portfolio, target_portfolio):
        """
        This function will estimate the transaction cost given the time, stock price, current portfolio and the target
        portfolio.
        """
        pass




class FlatFeeTransactionCostModel(TransactionCostModel):
    """
    This is arguably the simplest transaction cost model. The transaction cost is a fraction of the total amount
    of the trade. So this model will ignore the price information of the stocks.
    """

    def __str__(self):
        line_1 = "\nFlatFeeTransactionCostModel"
        line_2 = "\t{:<20}: {:>14.4f}".format('fraction', self._fraction)
        return '\n'.join([line_1, line_2])


    def __init__(self, fraction=0.005):
        self._fraction = fraction

    def estimate_cost(self, curr_t, priceMat, curr_portfolio, target_portfolio):
        return np.sum(np.abs(curr_portfolio - target_portfolio)) * self._fraction


    @property
    def fraction(self):
        return self._fraction

    @fraction.setter
    def fraction(self,value):
        self._fraction = value



