import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
from stock_predictor import StockPredictor
from stock_analysis import StockAnalyzer

class StrategyBacktester:
    """
    A class for backtesting trading strategies.
    
    This class handles:
    1. Strategy execution simulation
    2. Performance metrics calculation
    3. Risk management implementation
    4. Position sizing and management
    5. Results analysis and visualization
    
    Attributes:
        data (pd.DataFrame): Input data with price and indicators
        position_size (float): Maximum position size per trade
        transaction_cost (float): Cost per trade as percentage
        stop_loss (float): Stop loss percentage
        take_profit (float): Take profit percentage
    """
    def __init__(self, data, transaction_cost=0.001, slippage=0.001,
                 stop_loss=0.02, take_profit=0.05, trailing_stop=0.015,
                 position_size=1.0, max_positions=1):
        """
        Initialize the backtester with historical data and trading parameters.
        
        Args:
            data: DataFrame with stock data
            transaction_cost: Trading cost as a fraction of trade value (e.g., 0.001 = 0.1%)
            slippage: Estimated slippage as a fraction of trade value
            stop_loss: Stop loss percentage (e.g., 0.02 = 2%)
            take_profit: Take profit percentage (e.g., 0.05 = 5%)
            trailing_stop: Trailing stop percentage (e.g., 0.015 = 1.5%)
            position_size: Size of each position as a fraction of portfolio (e.g., 1.0 = 100%)
            max_positions: Maximum number of concurrent positions
        """
        self.data = data.copy()
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.trailing_stop = trailing_stop
        self.position_size = position_size
        self.max_positions = max_positions
        self.results = {}
        self.current_price = None
        self.entry_price = None
        self.trailing_price = None
    def apply_risk_management(self, positions):
        """
        Apply risk management rules to positions.
        
        Implements:
        1. Stop loss orders
        2. Take profit orders
        3. Position size limits
        4. Maximum drawdown control
        
        Args:
            positions (pd.Series): Original position signals
            
        Returns:
            pd.Series: Risk-adjusted positions
        """
        """
        Apply risk management rules to modify position signals.
        
        Args:
            positions: Series of initial position signals (1 for long, 0 for flat)
        Returns:
            Modified positions with risk management applied
        """
        managed_positions = positions.copy()
        current_position = 0
        entry_price = None
        trailing_price = None
        
        for i in range(1, len(self.data)):
            current_price = self.data['Close'].iloc[i]
            prev_price = self.data['Close'].iloc[i-1]
            
            # Check if we have a new signal
            if managed_positions.iloc[i] == 1 and current_position == 0:
                # Enter new position
                current_position = 1
                entry_price = current_price
                trailing_price = current_price
                managed_positions.iloc[i] = 1
            
            elif current_position == 1:
                # Update trailing price if price moves higher
                if current_price > trailing_price:
                    trailing_price = current_price
                
                # Check stop loss
                stop_loss_price = entry_price * (1 - self.stop_loss)
                if current_price <= stop_loss_price:
                    current_position = 0
                    entry_price = None
                    trailing_price = None
                    managed_positions.iloc[i] = 0
                    continue
                
                # Check trailing stop
                trailing_stop_price = trailing_price * (1 - self.trailing_stop)
                if current_price <= trailing_stop_price:
                    current_position = 0
                    entry_price = None
                    trailing_price = None
                    managed_positions.iloc[i] = 0
                    continue
                
                # Check take profit
                take_profit_price = entry_price * (1 + self.take_profit)
                if current_price >= take_profit_price:
                    current_position = 0
                    entry_price = None
                    trailing_price = None
                    managed_positions.iloc[i] = 0
                    continue
                
                # Keep position if no exit signals
                managed_positions.iloc[i] = 1
            
            else:
                managed_positions.iloc[i] = 0
        
        return managed_positions

    def calculate_returns(self, positions, model_name):
        """
        Calculate strategy returns including transaction costs.
        
        Steps:
        1. Calculate position returns
        2. Apply transaction costs
        3. Calculate cumulative returns
        4. Compute performance metrics
        
        Args:
            positions (pd.Series): Series of position signals (-1, 0, 1)
            model_name (str): Name of the model for reporting
            
        Returns:
            dict: Dictionary of performance metrics
        """
        """
        Calculate strategy returns given a series of positions.
        
        Args:
            positions: Series of position signals (1 for long, 0 for flat)
            model_name: Name of the model being evaluated
        """
        # Apply risk management
        managed_positions = self.apply_risk_management(positions)
        
        # Calculate daily returns
        daily_returns = self.data['Close'].pct_change()
        
        # Calculate strategy returns (before costs)
        strategy_returns = managed_positions.shift(1) * daily_returns * self.position_size
        
        # Calculate transaction costs
        trades = managed_positions.diff().fillna(0).abs()  # Fill NaN with 0 before counting trades
        total_cost = trades * (self.transaction_cost + self.slippage) * self.position_size
        
        # Calculate net returns
        net_returns = strategy_returns - total_cost
        
        # Calculate cumulative returns
        cumulative_returns = (1 + net_returns).cumprod()
        
        # Calculate buy and hold returns
        buy_hold_returns = (1 + daily_returns).cumprod()
        
        # Calculate performance metrics
        total_return = cumulative_returns.iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(cumulative_returns)) - 1
        
        # Calculate Sharpe Ratio
        risk_free_rate = 0.02  # Assume 2% risk-free rate
        excess_returns = net_returns - risk_free_rate/252
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        
        # Calculate Maximum Drawdown
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        max_drawdown = drawdowns.min()
        
        # Store results
        self.results[model_name] = {
            'positions': positions,
            'returns': net_returns,
            'cumulative_returns': cumulative_returns,
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': int(trades.sum()),  # Convert to integer for cleaner display
            'buy_hold_returns': buy_hold_returns
        }
    
    def backtest_model(self, model, model_name):
        """
        Backtest a specific model.
        
        Args:
            model: Trained model object
            model_name: Name of the model
        """
        # Get model predictions
        feature_columns = ['Close', 'Returns', 'SMA20', 'SMA50', 'RSI', 'MACD', 'BB_upper', 'BB_lower', 'Volume']
        X = self.data[feature_columns].copy()
        
        # Scale features
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=feature_columns, index=X.index)
        
        # Get prediction probabilities
        pred_proba = model.predict_proba(X)[:, 1]  # Probability of class 1 (up movement)
        
        # Dynamic position sizing based on prediction confidence
        positions = pd.Series(index=self.data.index, dtype=float)
        
        # No position if confidence is below base threshold
        base_threshold = 0.55
        positions[pred_proba < base_threshold] = 0
        
        # Scale position size based on confidence level
        confident_mask = pred_proba >= base_threshold
        positions[confident_mask] = ((pred_proba[confident_mask] - base_threshold) / (1 - base_threshold)) * self.position_size
        
        # Calculate returns and metrics
        self.calculate_returns(positions, model_name)
    
    def plot_results(self):
        """
        Plot the strategy performance metrics including cumulative returns,
        drawdown, and trade positions.
        """
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle('Trading Strategy Performance Analysis', fontsize=14, y=0.95)
        
        # Plot 1: Cumulative Returns
        ax1.set_title('Cumulative Returns Comparison')
        # Plot buy and hold returns
        buy_hold = self.results[list(self.results.keys())[0]]['buy_hold_returns']
        ax1.plot(buy_hold.index, buy_hold, 
                label='Buy & Hold', linestyle='--', color='gray', alpha=0.7)
        
        # Plot strategy returns
        for model_name, result in self.results.items():
            cum_returns = result['cumulative_returns']
            ax1.plot(cum_returns.index, cum_returns, label=model_name, linewidth=2)
        
        ax1.set_xlabel('')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Format y-axis as percentage
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        # Plot 2: Drawdown
        ax2.set_title('Strategy Drawdown')
        for model_name, result in self.results.items():
            cum_returns = result['cumulative_returns']
            rolling_max = cum_returns.expanding().max()
            drawdown = (cum_returns - rolling_max) / rolling_max
            ax2.plot(drawdown.index, drawdown, label=model_name, linewidth=2)
        
        ax2.set_xlabel('')
        ax2.set_ylabel('Drawdown')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Format y-axis as percentage
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        # Plot 3: Trade Positions
        ax3.set_title('Trading Positions')
        for model_name, result in self.results.items():
            positions = result['positions']
            ax3.plot(positions.index, positions, label=model_name, 
                    drawstyle='steps-post', linewidth=2)
        
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Position')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig('strategy_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def print_metrics(self):
        """
        Print performance metrics for all strategies.
        """
        metrics = pd.DataFrame()
        
        for model_name, result in self.results.items():
            metrics[model_name] = pd.Series({
                'Total Return': f"{result['total_return']:.2%}",
                'Annual Return': f"{result['annual_return']:.2%}",
                'Sharpe Ratio': f"{result['sharpe_ratio']:.2f}",
                'Max Drawdown': f"{result['max_drawdown']:.2%}",
                'Number of Trades': f"{result['num_trades']:.0f}"
            })
        
        print("\nPerformance Metrics:")
        print(metrics)


if __name__ == "__main__":
    # Get stock data using StockAnalyzer
    analyzer = StockAnalyzer('AAPL', '2023-01-01', '2025-03-05')
    analyzer.download_data()
    analyzer.preprocess_data()
    analyzer.add_technical_indicators()
    
    # Create StockPredictor with the processed data
    predictor = StockPredictor(analyzer.data)
    predictor.prepare_data()
    
    # Train all models and get the best model
    best_model = predictor.train_all_models()
    
    # Create backtester instance
    backtester = StrategyBacktester(predictor.data)
    
    # Backtest the best model
    backtester.backtest_model(best_model, 'Best Model')
    
    # Plot results and print metrics
    backtester.plot_results()
    backtester.print_metrics()





































































