

"""
Script: download_stock_price_data.py

This script will help to download stock price data from Yahoo finance. Once we have the data available, we can store
it in csv or pickle format.


After downloading the data from Yahoo, a data-cleaning process is required. Most of the time, the data has two dimensions,
stock name and stock price attributes such as date, open, low, high, close, adjClose and volume.

In our design, we want to have data grouped by attribute. For example, if the attribute of interest in closed price,
we will have a DataFrame called close. The index is the date, each column will represent a stock.


Attention:
    Double check the calculation of the stock return DataFrame. Do not use "future data" in the current calculation.

"""





import pandas as pd
import numpy as np
from pandas.io.data import DataReader
import datetime as dt
import os


# set the start date and end date and the list of stocks

print ('initializing...')

start_time = dt.datetime(2009,1,1)
end_time   = dt.datetime(2013,06,30)

stock_list = ['aapl','msft','mmm','ibm','jpm','wmt','yhoo','gps','ge','f']


# create data container
# this is a dictionary. The key is the ticker of the stock and the value is the
# dataframe that contains the historical information of that stock


print ('downloading historical data from yahoo finance ...')
data_container = dict()

for stock in stock_list :
    data_container[stock] = DataReader(stock, 'yahoo', start_time, end_time)



# Attention:
# ^^^^^^^^^
# need to add some check functionality here.
# This step is necessary because some of the stocks entered the market recently and for these
# stocks they do not have enough history








# Now create variables that is convenient for data analysis

# The variables are in the form of a matrix(dataframe). Each column
# represents a stock and the row index is the date.

# Note that the column name of the dataframe should have the same order as the stock_list

print( 'creating DataFrame...')

# create price and volume variables

open      = pd.DataFrame()
close     = pd.DataFrame()
low       = pd.DataFrame()
high      = pd.DataFrame()
volume    = pd.DataFrame()
adjClose  = pd.DataFrame()


for ticker in stock_list:
    open[ticker] = data_container[ticker]['Open']
    close[ticker] = data_container[ticker]['Close']
    low[ticker] = data_container[ticker]['Low']
    high[ticker] = data_container[ticker]['High']
    volume[ticker] = data_container[ticker]['Volume']
    adjClose[ticker] = data_container[ticker]['Adj Close']



# create return variables
nrow , ncol    = adjClose.shape
stReturn_pre   = adjClose.values[1:] / adjClose.values[:-1] - 1.
stReturn       = pd.DataFrame(np.zeros_like(adjClose),index=adjClose.index, columns=adjClose.columns )
stReturn[1:]   = stReturn_pre
stReturn.ix[0] = np.nan




# save dataframe to disk in a binary form


print ('saving DataFrame to disk...')

open.to_pickle     (path=r'../data/open_dataframe')
close.to_pickle    (path=r'../data/close_dataframe')
low.to_pickle      (path=r'../data/low_dataframe')
high.to_pickle     (path=r'../data/high_dataframe')
volume.to_pickle   (path=r'../data/volume_dataframe')
adjClose.to_pickle (path=r'../data/adjClose_dataframe')
stReturn.to_pickle (path=r'../data/stReturn_dataframe')




print ('Done.')













