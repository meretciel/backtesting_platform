

import pandas as pd
import urllib2
import urlparse
import re
import numpy as np

from os import path



data_dir = r'/Users/Ruikun/workspace/backtesting_platform_local/data'



# ========= Download Data From Stockpup.com =================

base_url = r'http://www.stockpup.com/data/'

# we first extract the list of stocks available on the website stockpup.com
# At the time of writing, there is a Download section on the right side of the home page.
# we will grab the list of stocks there

response = urllib2.urlopen(base_url)

list_of_stock_symbol = []
list_duplicates      = []

pattern = '\s+title=\"fundamental_data_excel_(\w+)\.csv\">'

for line in response.readlines():
    if line.find('csv') != -1:
        # print(line)
        res = re.findall(pattern, line)
        if res and res[0] in list_of_stock_symbol:
            print('{} already exist. This file will not be download and the existing one will be removed. Please download manually.'.format(res[0]))
            list_duplicates.append(res[0])
            continue

        res and list_of_stock_symbol.append(res[0])


# Here we remove the duplicates in the list of symbol
# For those symbols that have duplicates, we need to create the list manually.
list_of_stock_symbol = [x for x in list_of_stock_symbol if not x in list_duplicates]

# Now create the list manually for the symbols that have duplicates.
# The exact filename can be found via "inspect element"
print(list_duplicates)

# For now we just ignore these stocks. It is not clear what is going on with these stocks. There may be some
# mislabeling here.
pass




# download the csv files
# the csv files are load into pd.DataFrame. For each csv file, we add a column indicating the symbol of the stock
list_df_stock = []
list_failure  = []

suffix = '_quarterly_financial_data.csv'

for item in list_of_stock_symbol:
    filename = ''.join([item,suffix])
    target_url = urlparse.urljoin(base_url, filename)
    try:
        df = pd.read_csv(target_url)
        df['symbol'] = item
        list_df_stock.append(df)
    except Exception as e:
        print("Failed to download file for {}".format(item))
        print('Error message:', e)
        list_failure.append(item)


# print(list_failure)

# ========= Data Preparation =============

# concat all the dataframes in the list
df = pd.concat(list_df_stock, axis=0)

df.to_csv(path.join(data_dir, 'test_fundamental_data.csv'), index=False)

df.columns.name = 'fundamental_variable'
df = df.set_index(['Quarter end', 'symbol'])

df = df.stack()
df = df.unstack(level=1)

# In the initial csv files, None is used to represent missing data
df[df == 'None'] = np.nan

df.columns.name = None

# df.reset_index().to_csv(path.join(data_dir, 'fundamental_data_of_stocks_stacked.csv'), index=False)

# split the large dataframe based on the fundamental_variables
# now each DataFrame represent one fundamental variable about a bunch of stocks.

df = df.reset_index()

def f(df):
    var_name = df.iloc[0]['fundamental_variable']
    var_name = re.sub('[-\s\&\/]', '_', var_name)
    df.to_csv(path.join(data_dir, var_name + '.csv'), index=False)

df.groupby(by=['fundamental_variable']).apply(f)

# TODO: The dates of stocks data are not aligned.

