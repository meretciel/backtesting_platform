

from abc import ABCMeta, abstractmethod, abstractproperty

import numpy as np


import backtesting.setting as _bsetting




class PortfolioBase():
    """
    PortfolioConfig class is used to define the initial status of a portfolio. It contains information about
        (1) initial capital of the portfolio. This is in $ amount
        (2) maximum margin allowed for the portfolio. When we have short position in a portfolio, we are required to
            have a margin account and we are also required to keep the margin ratio below a certain level.
        (3) portfolio type. This describe the type of portfolio. The common types are market neutral portfolio, dollar
            neutral portfolio, long-only portfolio etc.
    """
    __metaclass__ = ABCMeta


    def __init__(self, initial_capital=100000., max_margin=0.5):
        self._initial_capital = initial_capital
        self._max_margin      = max_margin
        self._portfolio_type  = None


    @abstractmethod
    def normalize_weights(self, weights):
        """
        This function will normalize the input weights based on the portfolio type.
        """
        pass


    def get_initial_portfolio(self, weights):
        """
        This function will provide a initial portfolio allocation based on the initial capital


        """
        return self.normalize_weights(weights) * self.initial_capital


    @property
    def initial_capital(self):
        return self._initial_capital

    @property
    def max_margin(self):
       return self._max_margin

    @property
    def portfolio_type(self):
        if self._portfolio_type is None:
            raise ValueError("Portfolio type is missing.")

        return self._portfolio_type





# User defined Portfolio



class DollarNeutralPortfolio(PortfolioBase):

    def __init__(self, initial_capital=100000., max_margin=0.5):
        super(DollarNeutralPortfolio, self).__init__(initial_capital, max_margin)
        self._portfolio_type  = _bsetting.DOLLAR_NEUTRAL_PORTFOLIO


    def normalize_weights(self, weights):

        # We first subtract the mean of the weights, and the sum of the new array will be zero
        weights = weights - weights.mean()

        # Now separate the negative weights and positive weights
        neg_weights = weights[weights <= 0]
        pos_weights = weights[weights  >= 0]

        # In this step, we will normalize the negative weights(respectively positive weights) to 1
        # If the weights are too small, we will set them to zero
        neg_sum = neg_weights.sum()
        pos_sum = pos_weights.sum()

        if neg_sum.sum() < -0.0001:
            neg_weights = -neg_weights / neg_sum
        else:
            neg_weights = 0.

        if pos_sum > 0.0001:
            pos_weights = pos_weights / pos_sum
        else:
            pos_weights = 0.

        # combine the negative weights and positive weights
        weights[weights <= 0] = neg_weights
        weights[weights >= 0] = pos_weights

        # In this implementation, both the long and short position is equal to the initial capital.
        return weights




class LongShortPortfolio(PortfolioBase):
    def __init__(self, initial_capital=100000., max_margin=0.5):
        super(LongShortPortfolio, self).__init__(initial_capital, max_margin)
        self._portfolio_type  = _bsetting.LONG_SHORT_PORTFOLIO


    def normalize_weights(self, weights):

        weights_sum = weights.sum()

        # if the total weights is too small, we wil make all weights to zero
        if abs(weights_sum) < 0.005:
            return np.zeros_like(weights)
        return weights / weights_sum




#
# class LongOnlyPortfolioConfig(PortfolioBase):
#     def __init__(self, initial_capital=100000., max_margin=0.5):
#         super(LongOnlyPortfolioConfig, self).__init__(initial_capital=initial_capital, max_margin=max_margin)
#         self._portfolio_type  = _bsetting.LONG_ONLY_PORTFOLIO
#
#
#     def normalize_weights(self, weights):
#         # check the weights
#         weights_sum = weights.sum()
#         if abs(weights_sum) < 0.005:
#             return np.zeros_like(weights)
#         else:
#             return weights / weights_sum
#
#
#
#



if __name__ == '__main__':
    w = np.array([1.,2.,3.])
    dollar_neutral_portfolio = DollarNeutralPortfolio()
    long_short_portfolio     = LongShortPortfolio()
    print dollar_neutral_portfolio.normalize_weights(w)
    print long_short_portfolio.normalize_weights(w)




