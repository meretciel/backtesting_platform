
#!/usr/bin/python

import numpy as np
import pandas as pd


def computeMaxDrawDown(priceSerie, capital) :
    return (np.max(priceSerie) - np.min(priceSerie)) / float(capital)



def computeSmoothedReturn(priceSerie, period=5):
    avg_endPrice = np.mean(priceSerie[-period:])
    avg_startPrice = np.mean(priceSerie[:period])
    return avg_endPrice / avg_startPrice - 1.



def computeSharpeRatio(priceSerie, period=5) :
    smoothed_return = computeSmoothedReturn(priceSerie, period)
    retSerie = priceSerie[1:] / priceSerie[:-1] - 1.
    risk = np.std(retSerie)
    return smoothed_return / risk




def computeReturnArray(arr) :
    return (arr[1:] / arr[:-1] - 1.)



def computeStdOfPriceArray(priceArr):
    retArr = priceArr[1:] / priceArr[:-1] - 1.
    return np.std(retArr)

