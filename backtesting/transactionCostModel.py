
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
    def estimate_cost(self, curr_t, stock_price, curr_portfolio, target_portfolio):
        """
        This function will estimate the transaction cost given the time, stock price, current portfolio and the target
        portfolio.
        """
        pass




class FlatFeeTransactionModel(TransactionCostModel):

    def __init__(self, fraction=0.005):
        self._fraction = fraction

    def estimate_cost(self, curr_t, stock_price, curr_portfolio, target_portfolio):
        return np.sum(np.abs(curr_portfolio - target_portfolio) * self._fraction)


