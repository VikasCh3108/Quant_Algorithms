"""
Data Viewer Module
================

This module provides functionality to view and analyze raw stock price data.
It displays the data in both tabular and graphical formats for easy analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stock_analysis import StockAnalyzer

def display_raw_data(symbol='AAPL', start_date='2023-01-01', end_date='2025-03-05', display_rows=20):
    """
    Display raw stock price data in various formats.
    
    Args:
        symbol (str): Stock symbol
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        head_rows (int): Number of rows to display in preview
    """
    # Initialize analyzer and get data
    analyzer = StockAnalyzer(symbol, start_date, end_date)
    data = analyzer.download_data()
    
    if data is not None:
        print(f"\n{'='*50}")
        print(f"Raw Stock Price Data for {symbol}")
        print(f"{'='*50}")
        
        # Display first few rows of raw data
        # Display first rows
        print(f"\nFirst {display_rows} rows of raw data:")
        print(f"{'-'*50}")
        pd.set_option('display.max_columns', 7)
        pd.set_option('display.width', 120)
        print(data[['Open', 'High', 'Low', 'Close', 'Volume']].head(display_rows))
        
        # Display last rows
        print(f"\nLast {display_rows} rows of raw data:")
        print(f"{'-'*50}")
        print(data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(display_rows))
        
        # Display basic statistics
        print("\nBasic Statistics:")
        print(f"{'-'*30}")
        stats = data[['Open', 'High', 'Low', 'Close', 'Volume']].describe()
        pd.set_option('display.float_format', lambda x: '%.2f' % x)
        print(stats)
        
        # Calculate daily returns
        data['Daily_Return'] = data['Close'].pct_change()
        
        # Display key metrics
        print("\nKey Metrics:")
        print(f"{'-'*30}")
        print(f"Total Trading Days: {len(data)}")
        print(f"Average Daily Volume: {float(data['Volume'].mean()):,.0f}")
        print(f"Average Daily Return: {float(data['Daily_Return'].mean()):.2%}")
        print(f"Daily Return Std Dev: {float(data['Daily_Return'].std()):.2%}")
        print(f"Annual Volatility: {float(data['Daily_Return'].std() * np.sqrt(252)):.2%}")
        
        # Calculate monthly returns
        monthly_returns = data['Close'].resample('M').last().pct_change()
        print("\nMonthly Returns:")
        print("-" * 50)
        print(monthly_returns.tail(12))
        
        # Calculate quarterly returns
        quarterly_returns = data['Close'].resample('Q').last().pct_change()
        print("\nQuarterly Returns:")
        print("-" * 50)
        print(quarterly_returns.tail(4))
        
        # Print total return
        start_price = data['Close'].iloc[0]
        end_price = data['Close'].iloc[-1]
        total_return = (end_price - start_price) / start_price
        print(f"\nTotal Return: {total_return:.2%}")
        
        return data
    else:
        print(f"Error: Could not fetch data for {symbol}")
        return None

if __name__ == "__main__":
    # Display data for AAPL
    data = display_raw_data('AAPL')
