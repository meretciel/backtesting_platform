"""
Script: initialize_platform.py

This script will initialize the platform when it is download from the Github. User can modify the parameters defined
in this script. Please find more details in the comments.

To initialize the platform, please run this script in python.

"""


import os
from os import path
import pandas as pd

import script.download_stock_price_data as download_stock_price_data
import script.download_stock_fundamental_data as download_stock_fundamental_data

DATA_DIR          = 'data_2016_06_07'
BASIC_DATA_CENTER = 'data_2016_06_07/data_center'


# Specify the start and end date of the simulation period.
# The system will prepare the data between this range
START_DATE = pd.datetime(2009, 1, 1)
END_DATE   = pd.datetime(2013,06,30)

# Specify stocks of interests.
# If it is set None, the system will use the default value.
LIST_STOCKS = ['aapl','msft','mmm','ibm','jpm','wmt','yhoo','gps','ge','f']


if __name__ == '__main__':
    # create data directory.
    # The data directory will contain the data necessary for the simulation.
    curr_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = path.join(curr_dir, DATA_DIR)

    if path.exists(data_dir):
        print('Data directory exists. No data directory created.')
    else:
        os.mkdir(data_dir)
        print('Create: {}'.format(data_dir))

    # download and pre-processing the data

    # download data from yahoo finance. This is the stock price data
    print("Downloading stock price data from Yahoo finance")
    index = download_stock_price_data.download(start_date=START_DATE, end_date=END_DATE, list_stock=LIST_STOCKS, output_path=data_dir)
    # download data from stockpup.com. This is fundamental data of the stocks
    print("Downloading fundamental data of stocks from Stockpup.com")
    df = download_stock_fundamental_data.download(output_path=path.join(data_dir, 'fundamental_data_of_stocks.csv'))
    print("Preparing the fundamental variables.")
    download_stock_fundamental_data.prepare_and_save_data(df, index=index, output_path=data_dir)

    print("\n")
    print("************************")
    print("Initialization complete.")
    print("************************")

