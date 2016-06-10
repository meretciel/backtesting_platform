

import pandas as pd
import urllib2
import urlparse
import re
import numpy as np

from os import path



data_dir = r'/Users/Ruikun/workspace/backtesting_platform_local/data'



# ========= Download Data From Stockpup.com =================

def download(output_path=None):
    """
    This function will download fundamental data of stocks from stockpup.com. The results will be stored in a
    pd.DataFrame, which is returned by this function
    """
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


    df_out = pd.concat(list_df_stock, axis=0)

    if output_path is not None:
        df_out.to_csv(output_path, index=False)
    return df_out




# ========= Data Preparation =============

def prepare_and_save_data(df, index=None, output_path=None):
    """
    This function will prepare the download data to fit the platform.
    :return:
    """
    assert index is not None
    assert output_path is not None

    df.columns.name = 'fundamental_variable'
    df = df.set_index(['Quarter end', 'symbol'])

    df = df.stack()
    df = df.unstack(level=1)

    # In the initial csv files, None is used to represent missing data
    df[df == 'None'] = np.nan

    df.columns.name = None

    #df.reset_index().to_csv(path.join(data_dir, 'fundamental_data_of_stocks_stacked.csv'), index=False)

    # split the large dataframe based on the fundamental_variables
    # now each DataFrame represent one fundamental variable about a bunch of stocks.
    df = df.reset_index()


    # convert Quarter end to datetime object
    df['Quarter end'] = pd.to_datetime(df['Quarter end'])
    df = df.sort(['Quarter end'], axis=0)
    # convert the strings to float
    df = df.set_index(['Quarter end', 'fundamental_variable']).astype(float).reset_index(level=1)


    def _reindex(df):
        # This function will be applied to each fundamental variable group.
        # It will reindex the DataFrame. It will add the target index to the existing index and sort the new index
        new_index = index.append(df.index).drop_duplicates().values
        new_index.sort()

        # fill NaN with forward fill method. We should use forward filling because we can only use historical data
        # not the data from the future.
        df = df.reindex(new_index).fillna(method='ffill')

        return df


    def _to_csv(df):
        var_name = df.iloc[0]['fundamental_variable']
        print("{:<30} {}".format("preparing fundamental variable", var_name))
        var_name = re.sub('[-\s\&\/]', '_', var_name)           # remove special characters
        var_name = re.sub('_+', '_', var_name)                  # remove repetitive underscores
        var_name = re.sub('_$', '', var_name)                   # remove the tailing underscore
        df.set_index(['Quarter end']).reindex(index).to_csv(path.join(output_path, var_name + '.csv'), index=True)



    # The following step is to reindex the DataFrame. Note that the reindex is done by each fundamental variable.
    # The reason for this is if we do the reindex without groupby, there will be NaN in the fundamental_variable after
    # reindex operation, which is problematic for the forward fill operation of the next step
    df_reindex = df.groupby(by=['fundamental_variable'], as_index=False).apply(_reindex).sort_index()
    df_reindex.index = df_reindex.index.droplevel(0)

    # save to csv file by each fundamental variable
    df_out = df_reindex.reset_index()
    df_out.groupby(by=['fundamental_variable']).apply(_to_csv)




if __name__ == '__main__':
    df = download(output_path=path.join(data_dir, 'fundamental_data_of_stocks.csv'))
    # prepare_and_save_data(index=None, output_path=None)






#
#
# df.groupby(by=['fundamental_variable']).apply(f)
#
#
#
# df = pd.read_csv(path.join(data_dir, 'fundamental_data_of_stocks_stacked.csv'))
#
#
# df_out = df.groupby(by=['fundamental_variable'], as_index=False).ffill()
#
#
# df_out.to_csv(path.join(data_dir, 'fundamental_data_of_stocks_stacked_filled.csv'), index=False)
#
#
#
# df_ref = pd.read_pickle(path.join(data_dir, 'high_dataframe'))
#
# df_out = df_out.set_index(['Quarter end'])
#
# df_out = df_out.reindex(df_ref.index)
#
#
# def g(df):
#     val = df.iloc[0]['fundamental_variable']
#     df = df.reindex(df_ref.index)
#     df['fundamental_variable'] = val
#     return df
#
# df_out2 = df_out.groupby(by=['fundamental_variable'], as_index=False).apply(g).sort_index()
# df_out2.index = df_out2.index.droplevel(0)
#
# df_out3 = df_out2.groupby(by=['fundamental_variable'], as_index=False).ffill().reset_index()
#
#
#
#
# df_out3.to_csv(path.join(data_dir, 'fundamental_data_of_stocks_stacked_filled_truncated.csv'), index=False)
#
#
#
# def f(df):
#     var_name = df.iloc[0]['fundamental_variable']
#     var_name = re.sub('[-\s\&\/]', '_', var_name)
#     df.to_csv(path.join(data_dir, var_name + '.csv'), index=False)
#
# df_out3.groupby(by=['fundamental_variable']).apply(f)
#
#
#
