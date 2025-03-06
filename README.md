# Algorithmic Trading Strategy with Machine Learning

## Project Overview
This project implements a comprehensive algorithmic trading strategy using both traditional machine learning and deep learning approaches. It includes feature engineering, model training, backtesting, and performance evaluation.

## Project Structure
```
quant_algorithm/
├── stock_analysis.py       # Stock data download and technical indicators
├── stock_predictor.py      # Traditional ML models (RF, XGBoost, LogReg)
├── deep_learning_models.py # Deep learning models (LSTM, GRU)
├── strategy_backtester.py  # Strategy backtesting and evaluation
├── model_optimizer.py      # Hyperparameter optimization
└── requirements.txt        # Project dependencies
```

## Features
1. **Technical Indicators**
   - Moving Averages (SMA, EMA)
   - RSI (Relative Strength Index)
   - MACD (Moving Average Convergence Divergence)
   - Bollinger Bands
   - Volume Indicators

2. **Machine Learning Models**
   - Random Forest (High precision: 80%)
   - XGBoost (Balanced performance: F1=55.12%)
   - Logistic Regression (Baseline model)
   - LSTM (Validation accuracy: 65.12%)
   - GRU (Test accuracy: 54.21%)

3. **Risk Management**
   - Dynamic position sizing
   - Stop-loss implementation
   - Volatility-based adjustments

## Installation
1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preparation
```python
from stock_analysis import StockAnalyzer

# Initialize analyzer
analyzer = StockAnalyzer('AAPL', '2023-01-01', '2025-03-05')
analyzer.download_data()
analyzer.add_technical_indicators()
```

### 2. Traditional ML Models
```python
from stock_predictor import StockPredictor

# Train models
predictor = StockPredictor(analyzer.data)
predictor.prepare_data()
best_model = predictor.train_all_models()
```

### 3. Deep Learning Models
```python
from deep_learning_models import DeepLearningPredictor

# Train deep learning models
dl_predictor = DeepLearningPredictor(analyzer.data)
X_train, X_test, y_train, y_test = dl_predictor.prepare_data()

# Train LSTM
lstm_model = dl_predictor.build_lstm_model()
lstm_model, history = dl_predictor.train_model(lstm_model, "LSTM")

# Train GRU
gru_model = dl_predictor.build_gru_model()
gru_model, history = dl_predictor.train_model(gru_model, "GRU")
```

### 4. Backtesting
```python
from strategy_backtester import StrategyBacktester

# Run backtest
backtester = StrategyBacktester(analyzer.data)
results = backtester.run_backtest(model)
```

## Model Performance Summary

### Traditional Models
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|---------|----------|
| Random Forest | 50.46% | 80.00% | 13.33% | 22.86% |
| XGBoost | 47.71% | 52.24% | 58.33% | 55.12% |
| Logistic Regression | 44.95% | 0.00% | 0.00% | 0.00% |

### Deep Learning Models
| Model | Train Accuracy | Test/Val Accuracy |
|-------|---------------|-------------------|
| LSTM | 51.15% | 65.12% |
| GRU | 55.40% | 54.21% |

## Best Practices

1. **Feature Selection**
   - Use consistent features across all models
   - Focus on proven technical indicators
   - Handle missing values appropriately

2. **Model Selection**
   - For high precision: Use Random Forest
   - For balanced performance: Use GRU or XGBoost
   - For trend following: Use LSTM

3. **Risk Management**
   - Implement stop-loss orders
   - Use position sizing based on volatility
   - Consider market regime in decision making

## Contributing
Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
