

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
from pandas.io.data import DataReader
from os import path



# set the start date and end date and the list of stocks
#
# start_date = pd.datetime(2009,1,1)
# end_date   = pd.datetime(2013,06,30)
# stock_list = ['aapl','msft','mmm','ibm','jpm','wmt','yhoo','gps','ge','f']
#
#
# df = DataReader('AAPL', 'yahoo', start_date, end_date)
#
# df.to_csv(path.join(r'/Users/Ruikun/workspace/backtesting_platform_local/data', 'test_index.csv'))

_default_list_stock = ['aapl','msft','mmm','ibm','jpm','wmt','yhoo','gps','ge','f']


def download(start_date=None, end_date=None, list_stock=None, output_path=None):
    """
    This function will download data from Yahoo Finance.

    Args:
        start_date:    datetime object indicating the start of the period
        end_date:      datetime object indicating the end of the period
        list_of_stock: list of stocks of interest. The stock is identified by symbol.
        output_path:   path to store the download files.

    Return:
        The function will return the Date index of the download files. This is useful when we need to reindex other
        pd.DataFrame.

    """
    assert start_date
    assert end_date
    assert output_path

    list_stock = list_stock or _default_list_stock
    list_stock = [x.upper() for x in list_stock]

    # create data container
    # this is a dictionary. The key is the ticker of the stock and the value is the
    # DateFrame that contains the historical information of that stock
    data_container = dict()

    for stock in list_stock :
        try:
            data_container[stock] = DataReader(stock, 'yahoo', start_date, end_date)
        except Exception as e:
            print("Failed to download {} price data from Yahoo Finance.".format(stock))
            print(e)

    open      = pd.DataFrame()
    close     = pd.DataFrame()
    low       = pd.DataFrame()
    high      = pd.DataFrame()
    volume    = pd.DataFrame()
    adjClose  = pd.DataFrame()


    for ticker in list_stock:
        open[ticker]     = data_container[ticker]['Open']
        close[ticker]    = data_container[ticker]['Close']
        low[ticker]      = data_container[ticker]['Low']
        high[ticker]     = data_container[ticker]['High']
        volume[ticker]   = data_container[ticker]['Volume']
        adjClose[ticker] = data_container[ticker]['Adj Close']


    open.to_csv(path.join(output_path,         'open.csv'))
    close.to_csv(path.join(output_path,       'close.csv'))
    low.to_csv(path.join(output_path,           'low.csv'))
    high.to_csv(path.join(output_path,         'high.csv'))
    volume.to_csv(path.join(output_path,     'volume.csv'))
    adjClose.to_csv(path.join(output_path, 'ajdClose.csv'))


    rtn = adjClose / adjClose.shift(1) - 1.
    rtn.to_csv(path.join(output_path, 'rtn.csv'))

    return volume.index











# Now create variables that is convenient for data analysis

# The variables are in the form of a matrix(dataframe). Each column
# represents a stock and the row index is the date.

# Note that the column name of the dataframe should have the same order as the stock_list



# create price and volume variables


#
#
# # create return variables
# nrow , ncol    = adjClose.shape
# stReturn_pre   = adjClose.values[1:] / adjClose.values[:-1] - 1.
# stReturn       = pd.DataFrame(np.zeros_like(adjClose),index=adjClose.index, columns=adjClose.columns )
# stReturn[1:]   = stReturn_pre
# stReturn.ix[0] = np.nan
#
#
#
#
# # save dataframe to disk in a binary form
#
#
# print ('saving DataFrame to disk...')
#
# open.to_pickle     (path=r'../data/open_dataframe')
# close.to_pickle    (path=r'../data/close_dataframe')
# low.to_pickle      (path=r'../data/low_dataframe')
# high.to_pickle     (path=r'../data/high_dataframe')
# volume.to_pickle   (path=r'../data/volume_dataframe')
# adjClose.to_pickle (path=r'../data/adjClose_dataframe')
# stReturn.to_pickle (path=r'../data/stReturn_dataframe')
#
#
#
#
# print ('Done.')
#
#











