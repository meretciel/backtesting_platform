
#!/usr/bin/python

import numpy as np
import pandas as pd




def computeMaxDrawDown(priceSeries, capital):
    size = len(priceSeries)
    arr_min = np.empty(size)

    tmp_min = np.inf
    for i in xrange(size):
        pos = size-1-i
        val = priceSeries[pos]
        if (val < tmp_min) :
            tmp_min = val
        arr_min[pos] = tmp_min

    return np.max(priceSeries - arr_min) / float(capital)









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

