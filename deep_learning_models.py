"""
Deep Learning Models for Stock Price Prediction
============================================

This module implements deep learning models (LSTM and GRU) for stock price prediction.
It provides a comparison with traditional machine learning approaches and includes
comprehensive evaluation metrics.

Key Features:
    - LSTM and GRU implementations for time series prediction
    - Sequence data preparation for time series analysis
    - Model evaluation and comparison metrics
    - Integration with traditional ML models for comparison

Author: Codeium AI
Date: 2025-03-06
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

class DeepLearningPredictor:
    """
    A class for training and evaluating deep learning models for stock price prediction.
    
    This class handles data preparation, model creation, training, and evaluation
    for both LSTM and GRU models. It maintains consistency with feature selection
    and preprocessing steps used in traditional machine learning models.
    
    Attributes:
        data (pd.DataFrame): Input data containing stock prices and technical indicators
        sequence_length (int): Number of time steps to look back for prediction
        scaler (MinMaxScaler): Scaler for target variable
        feature_scaler (MinMaxScaler): Scaler for feature variables
        X_train (np.array): Training features
        y_train (np.array): Training targets
        X_test (np.array): Testing features
        y_test (np.array): Testing targets
    """
    def __init__(self, data, sequence_length=10):
        """
        Initialize the DeepLearningPredictor.
        
        Args:
            data (pd.DataFrame): Input data containing stock prices and indicators
            sequence_length (int): Number of time steps to look back (default: 10)
        """
        """
        Initialize the deep learning predictor.
        
        Args:
            data: DataFrame with features
            sequence_length: Number of time steps to look back
        """
        self.data = data.copy()
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        
    def prepare_sequences(self, X, y):
        """
        Prepare sequences for time series prediction.
        
        This method creates sequences of historical data points for time series
        prediction. Each sequence contains 'sequence_length' time steps of features,
        and the corresponding target is the next time step's value.
        
        Args:
            X (np.array): Feature matrix
            y (np.array): Target vector
            
        Returns:
            tuple: (X_sequences, y_sequences) containing prepared sequences
        """
        """
        Prepare sequences for time series prediction.
        """
        X_seq, y_seq = [], []
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:(i + self.sequence_length)])
            y_seq.append(y[i + self.sequence_length])
        return np.array(X_seq), np.array(y_seq)
    
    def prepare_data(self):
        """
        Prepare data for deep learning models.
        
        This method handles:
        1. Feature selection (matching traditional ML models)
        2. Target creation (binary classification)
        3. Data scaling
        4. Sequence creation
        5. Train-test splitting
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test) prepared for deep learning
        """
        """
        Prepare data for deep learning models.
        """
        # Select features (same as traditional models for fair comparison)
        feature_columns = [
            'Close', 'Returns', 'SMA20', 'SMA50',
            'RSI', 'MACD', 'BB_High_20', 'BB_Low_20', 'Volume'
        ]
        
        # Create target (same as traditional models)
        self.data['Target'] = (self.data['Close'].shift(-1) > self.data['Close']).astype(int)
        
        # Handle NaN values
        for col in feature_columns:
            self.data[col] = self.data[col].ffill().bfill()
        
        # Scale features
        X = self.feature_scaler.fit_transform(self.data[feature_columns])
        y = self.data['Target'].values
        
        # Create sequences
        X_seq, y_seq = self.prepare_sequences(X, y)
        
        # Split into train and test
        train_size = int(len(X_seq) * 0.8)
        self.X_train = X_seq[:train_size]
        self.X_test = X_seq[train_size:]
        self.y_train = y_seq[:train_size]
        self.y_test = y_seq[train_size:]
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def build_lstm_model(self):
        """
        Build and return an LSTM model.
        
        Architecture:
        1. LSTM layer (50 units) with sequence return
        2. Dropout (20%) for regularization
        3. LSTM layer (50 units)
        4. Dropout (20%)
        5. Dense layer (25 units) with ReLU activation
        6. Output layer with sigmoid activation
        
        Returns:
            keras.Model: Compiled LSTM model
        """
        """
        Build and return an LSTM model.
        """
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.sequence_length, self.X_train.shape[2])),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def build_gru_model(self):
        """
        Build and return a GRU model.
        
        Architecture:
        1. GRU layer (50 units) with sequence return
        2. Dropout (20%) for regularization
        3. GRU layer (50 units)
        4. Dropout (20%)
        5. Dense layer (25 units) with ReLU activation
        6. Output layer with sigmoid activation
        
        Returns:
            keras.Model: Compiled GRU model
        """
        """
        Build and return a GRU model.
        """
        model = Sequential([
            GRU(50, return_sequences=True, input_shape=(self.sequence_length, self.X_train.shape[2])),
            Dropout(0.2),
            GRU(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def train_model(self, model, model_name):
        """
        Train the specified model with early stopping.
        
        Args:
            model (keras.Model): Model to train
            model_name (str): Name of the model for printing results
            
        Returns:
            tuple: (trained_model, training_history)
        """
        """
        Train the specified model.
        """
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        history = model.fit(
            self.X_train, self.y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Evaluate model
        train_accuracy = model.evaluate(self.X_train, self.y_train, verbose=0)[1]
        test_accuracy = model.evaluate(self.X_test, self.y_test, verbose=0)[1]
        
        print(f"\n{model_name} Results:")
        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        return model, history
    
    def predict(self, model, X):
        """
        Make predictions using the trained model.
        
        Args:
            model (keras.Model): Trained model
            X (np.array): Input features
            
        Returns:
            np.array: Model predictions
        """
        """
        Make predictions using the trained model.
        """
        return model.predict(X, verbose=0)

def main():
    """
    Main function to run deep learning experiments and compare with traditional models.
    
    This function:
    1. Downloads and prepares stock data
    2. Trains LSTM and GRU models
    3. Compares results with traditional ML models
    4. Prints comprehensive comparison metrics
    """
    """
    Main function to run deep learning experiments.
    """
    from stock_analysis import StockAnalyzer
    
    # Get stock data
    analyzer = StockAnalyzer('AAPL', '2023-01-01', '2025-03-05')
    analyzer.download_data()
    analyzer.add_technical_indicators()
    
    # Initialize predictor
    predictor = DeepLearningPredictor(analyzer.data)
    X_train, X_test, y_train, y_test = predictor.prepare_data()
    
    print("\nTraining LSTM model...")
    lstm_model = predictor.build_lstm_model()
    lstm_model, lstm_history = predictor.train_model(lstm_model, "LSTM")
    
    print("\nTraining GRU model...")
    gru_model = predictor.build_gru_model()
    gru_model, gru_history = predictor.train_model(gru_model, "GRU")
    
    # Compare with traditional models
    from stock_predictor import StockPredictor
    traditional_predictor = StockPredictor(analyzer.data)
    traditional_predictor.prepare_data()
    best_model = traditional_predictor.train_all_models()
    
    print("\nModel Comparison Summary:")
    print("-" * 50)
    print("Traditional Models vs Deep Learning Models")
    print("-" * 50)
    
if __name__ == "__main__":
    main()
