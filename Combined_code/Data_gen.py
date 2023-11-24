# Jupyter Code Block
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Function to fetch financial data
def fetch_financial_data(tickers, start_date='2003-01-01', end_date='2022-12-31'):
    data = yf.download(tickers, start=start_date, end=end_date)
    return data['Close']

# Data cleaning: fill NaN values using forward fill method
def fill_leading_nans_with_one(series):
    first_valid = series.first_valid_index()
    if first_valid is not None:
        series[:first_valid] = 1
    return series

def collect_data(tickers):

    data = fetch_financial_data(tickers)
    data_filled = data.fillna(method='ffill')
    if isinstance(data_filled, pd.DataFrame):
            
        cols = ['^GSPC'] + [col for col in data_filled if col != '^GSPC']
        data_filled = data_filled[cols]
        data_filled.columns = range(len(data_filled.columns))
    # data_filled = data_filled.pct_change(1)
    return data_filled