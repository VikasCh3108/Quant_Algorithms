"""
Stock Analysis Module
===================

This module handles stock data downloading and technical indicator calculations.
It provides a comprehensive set of technical analysis tools and data preprocessing
functionalities for algorithmic trading.

Features:
    - Data download from Yahoo Finance
    - Technical indicator calculations
    - Data preprocessing and cleaning
    - Moving averages and momentum indicators
    - Volatility and volume analysis

Author: Codeium AI
Date: 2025-03-06
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

class StockAnalyzer:
    """
    A class for downloading and analyzing stock data.
    
    This class handles the downloading of stock data from Yahoo Finance and
    calculates various technical indicators used in trading strategies.
    It includes methods for data preprocessing, technical analysis, and
    feature engineering.
    
    Attributes:
        symbol (str): Stock symbol (e.g., 'AAPL')
        start_date (str): Start date for data download (YYYY-MM-DD)
        end_date (str): End date for data download (YYYY-MM-DD)
        data (pd.DataFrame): Downloaded stock data with calculated indicators
    """
    def __init__(self, symbol, start_date, end_date):
        """
        Initialize the StockAnalyzer.
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL')
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        
    def download_data(self):
        """
        Download stock data from Yahoo Finance.
        
        Downloads OHLCV (Open, High, Low, Close, Volume) data for the specified
        stock symbol and date range. Handles missing data and ensures data quality.
        
        Returns:
            pd.DataFrame: Downloaded stock data
        """
        """Download stock data from Yahoo Finance"""
        try:
            self.data = yf.download(self.symbol, start=self.start_date, end=self.end_date)
            print(f"Successfully downloaded data for {self.symbol}")
            self.preprocess_data()
            self.add_technical_indicators()
            return self.data
        except Exception as e:
            print(f"Error downloading data: {e}")
            return None
            
    def preprocess_data(self):
        """Handle missing values and calculate returns"""
        if self.data is None:
            return False
            
        # Forward fill missing values
        self.data = self.data.ffill()
        
        # Calculate daily returns
        self.data['Returns'] = self.data['Close'].pct_change()
        
        # Remove outliers from returns (using 3 standard deviations)
        returns_mean = self.data['Returns'].mean()
        returns_std = self.data['Returns'].std()
        self.data['Returns'] = self.data['Returns'].clip(
            lower=returns_mean - 3*returns_std,
            upper=returns_mean + 3*returns_std
        )
        
        return True
        
    def add_technical_indicators(self):
        """
        Calculate and add technical indicators to the dataset.
        
        Calculates the following indicators:
        1. Price Indicators:
           - Returns and Log Returns
           - Price Rate of Change
        
        2. Moving Averages:
           - Simple Moving Averages (10, 20, 50, 100 days)
           - Exponential Moving Averages (10, 20, 50, 100 days)
        
        3. Momentum Indicators:
           - Relative Strength Index (RSI)
           - MACD (Moving Average Convergence Divergence)
        
        4. Volatility Indicators:
           - Bollinger Bands (20 and 50 days)
           - Average True Range (ATR)
           - Price Channels
        
        5. Volume Indicators:
           - Volume Moving Average
           - Volume Ratio
        
        Returns:
            None: Modifies the data DataFrame in place
        """
        """Add various technical indicators"""
        if self.data is None:
            return False
            
        # Price-based indicators
        self.data['Log_Returns'] = np.log(self.data['Close'] / self.data['Close'].shift(1))
        self.data['Price_Rate_Change'] = (self.data['Close'] - self.data['Close'].shift(5)) / self.data['Close'].shift(5)
        
        # Moving Averages
        for period in [10, 20, 50, 100]:
            # Use min_periods=1 to start calculating as soon as possible
            self.data[f'SMA{period}'] = self.data['Close'].rolling(window=period, min_periods=1).mean()
            self.data[f'EMA{period}'] = self.data['Close'].ewm(span=period, adjust=False, min_periods=1).mean()
        
        # Momentum Indicators
        # RSI
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = self.data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = self.data['Close'].ewm(span=26, adjust=False).mean()
        self.data['MACD'] = exp1 - exp2
        self.data['MACD_Signal'] = self.data['MACD'].ewm(span=9, adjust=False).mean()
        self.data['MACD_Hist'] = self.data['MACD'] - self.data['MACD_Signal']
        
        # Bollinger Bands
        for period in [20, 50]:
            sma = self.data['Close'].rolling(window=period, min_periods=1).mean()
            std = self.data['Close'].rolling(window=period, min_periods=1).std()
            bb_high = sma + (std * 2)
            bb_low = sma - (std * 2)
            
            self.data[f'BB_Middle_{period}'] = sma
            self.data[f'BB_High_{period}'] = bb_high
            self.data[f'BB_Low_{period}'] = bb_low
            
            # Calculate BB width and handle potential division by zero
            bb_width = ((bb_high.values - bb_low.values) / sma.values).flatten()
            bb_width = np.where(np.isinf(bb_width), np.nan, bb_width)
            bb_width = np.nan_to_num(bb_width, nan=0.0)
            self.data[f'BB_Width_{period}'] = bb_width
        
        # Volatility Indicators
        self.data['ATR'] = self.calculate_atr(14)
        self.data['Volatility'] = self.data['Returns'].rolling(window=20).std()
        
        # Volume Indicators
        volume_sma = self.data['Volume'].rolling(window=20, min_periods=1).mean()
        self.data['Volume_SMA20'] = volume_sma
        volume_ratio = (self.data['Volume'].values / volume_sma.values).flatten()
        # Replace inf values that occur when Volume_SMA20 is 0
        volume_ratio = np.where(np.isinf(volume_ratio), np.nan, volume_ratio)
        # Fill NaN with 1 (neutral value) for Volume_Ratio
        volume_ratio = np.nan_to_num(volume_ratio, nan=1.0)
        self.data['Volume_Ratio'] = volume_ratio
        
        # Price Channels
        for period in [20, 50]:
            high_period = self.data['High'].rolling(window=period, min_periods=1).max()
            low_period = self.data['Low'].rolling(window=period, min_periods=1).min()
            self.data[f'High_{period}'] = high_period
            self.data[f'Low_{period}'] = low_period
            
            # Calculate channel width and handle potential division by zero
            channel_width = ((high_period.values - low_period.values) / self.data['Close'].values).flatten()
            channel_width = np.where(np.isinf(channel_width), np.nan, channel_width)
            channel_width = np.nan_to_num(channel_width, nan=0.0)
            self.data[f'Channel_Width_{period}'] = channel_width
        
        return True
        
    def calculate_atr(self, period):
        """Calculate Average True Range (ATR)"""
        high_low = self.data['High'] - self.data['Low']
        high_close = np.abs(self.data['High'] - self.data['Close'].shift())
        low_close = np.abs(self.data['Low'] - self.data['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        return true_range.rolling(window=period).mean()
            
        close_series = self.data['Close']
        
        # Add Simple Moving Averages
        self.data['SMA20'] = close_series.rolling(window=20).mean()
        self.data['SMA50'] = close_series.rolling(window=50).mean()
        
        # Add RSI
        delta = close_series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))
        
        # Add MACD
        exp1 = close_series.ewm(span=12, adjust=False).mean()
        exp2 = close_series.ewm(span=26, adjust=False).mean()
        self.data['MACD'] = exp1 - exp2
        self.data['MACD_Signal'] = self.data['MACD'].ewm(span=9, adjust=False).mean()
        
        # Add Bollinger Bands
        rolling_mean = close_series.rolling(window=20).mean()
        rolling_std = close_series.rolling(window=20).std()
        self.data['BB_Middle'] = rolling_mean
        self.data['BB_High'] = rolling_mean + (rolling_std * 2)
        self.data['BB_Low'] = rolling_mean - (rolling_std * 2)
        
        return True
        
    def plot_analysis(self):
        """Create visualization plots"""
        if self.data is None:
            return False
            
        # Plot 1: Price and Technical Indicators
        plt.figure(figsize=(15, 10))
        plt.plot(self.data.index, self.data['Close'], label='Close Price')
        plt.plot(self.data.index, self.data['SMA20'], label='SMA 20')
        plt.plot(self.data.index, self.data['SMA50'], label='SMA 50')
        plt.plot(self.data.index, self.data['BB_High'], 'r--', label='BB High')
        plt.plot(self.data.index, self.data['BB_Low'], 'r--', label='BB Low')
        plt.title(f'{self.symbol} Price and Technical Indicators')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('price_indicators.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: RSI
        plt.figure(figsize=(15, 5))
        plt.plot(self.data.index, self.data['RSI'], color='purple')
        plt.axhline(y=70, color='r', linestyle='--')
        plt.axhline(y=30, color='g', linestyle='--')
        plt.title('RSI Indicator')
        plt.xlabel('Date')
        plt.ylabel('RSI')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('rsi.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 3: MACD
        plt.figure(figsize=(15, 5))
        plt.plot(self.data.index, self.data['MACD'], label='MACD', color='blue')
        plt.plot(self.data.index, self.data['MACD_Signal'], label='Signal Line', color='orange')
        plt.title('MACD')
        plt.xlabel('Date')
        plt.ylabel('MACD')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('macd.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 4: Returns Distribution
        plt.figure(figsize=(10, 5))
        plt.hist(self.data['Returns'].dropna(), bins=50, color='skyblue')
        plt.title('Returns Distribution')
        plt.xlabel('Returns')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig('returns.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 5: Volume
        plt.figure(figsize=(15, 5))
        plt.plot(self.data.index, self.data['Volume'], color='green')
        plt.title(f'{self.symbol} Trading Volume Over Time')
        plt.xlabel('Date')
        plt.ylabel('Volume')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('volume.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return True
        
        return True

def main():
    # Set parameters
    symbol = "AAPL"  # Example: Apple Inc.
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3*365)  # 3 years of data
    
    # Initialize and run analysis
    analyzer = StockAnalyzer(symbol, start_date, end_date)
    
    if not analyzer.download_data():
        print("Failed to download data. Exiting...")
        return
        
    analyzer.preprocess_data()
    analyzer.add_technical_indicators()
    
    # Display the first few rows of processed data
    print("\nFirst few rows of processed data:")
    print(analyzer.data[['Close', 'Returns', 'SMA20', 'SMA50', 'RSI', 'MACD', 'BB_High', 'BB_Low']].head())
    
    # Display basic statistics of the features
    print("\nBasic statistics of the features:")
    print(analyzer.data[['Close', 'Returns', 'SMA20', 'SMA50', 'RSI', 'MACD']].describe())
    
    # Check for any remaining missing values
    missing_values = analyzer.data[['Close', 'Returns', 'SMA20', 'SMA50', 'RSI', 'MACD', 'BB_High', 'BB_Low']].isnull().sum()
    print("\nMissing values in each column:")
    print(missing_values)
    
    analyzer.plot_analysis()
    
    print("\nAnalysis complete! Check the generated PNG files for visualizations.")

if __name__ == "__main__":
    main()
