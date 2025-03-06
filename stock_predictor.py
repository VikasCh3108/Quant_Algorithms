import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from stock_analysis import StockAnalyzer
from datetime import datetime, timedelta

class StockPredictor:
    """
    A class for training and evaluating machine learning models for stock prediction.
    
    This class handles:
    1. Data preprocessing and feature scaling
    2. Model training and hyperparameter optimization
    3. Model evaluation and comparison
    4. Prediction generation
    
    Attributes:
        data (pd.DataFrame): Input data with technical indicators
        X (np.array): Feature matrix
        y (np.array): Target vector
        X_train (np.array): Training features
        X_test (np.array): Testing features
        y_train (np.array): Training targets
        y_test (np.array): Testing targets
        scaler (StandardScaler): Feature scaler
        best_model: Best performing model
    """
    def __init__(self, data):
        """
        Initialize the StockPredictor.
        
        Args:
            data (pd.DataFrame): Input data containing price and technical indicators
        """
        # Convert data to DataFrame if it's not already
        self.data = data.copy()
        # If we have a MultiIndex, select the first symbol
        if isinstance(self.data.columns, pd.MultiIndex):
            symbol = self.data.columns.get_level_values(1)[0]
            self.data.columns = self.data.columns.get_level_values(0)
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.current_model = None
        
    def calculate_technical_indicators(self):
        # Calculate daily returns
        self.data['Returns'] = self.data['Close'].pct_change()
        
        # Calculate SMAs
        self.data['SMA20'] = self.data['Close'].rolling(window=20).mean()
        self.data['SMA50'] = self.data['Close'].rolling(window=50).mean()
        
        # Calculate RSI
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        exp1 = self.data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = self.data['Close'].ewm(span=26, adjust=False).mean()
        self.data['MACD'] = exp1 - exp2
        
        # Calculate Bollinger Bands
        sma20 = self.data['Close'].rolling(window=20).mean()
        std = self.data['Close'].rolling(window=20).std()
        self.data['BB_upper'] = sma20 + (std * 2)
        self.data['BB_lower'] = sma20 - (std * 2)
    
    def prepare_data(self):
        """
        Prepare data for machine learning models.
        
        Steps:
        1. Select relevant features
        2. Create target variable (binary classification)
        3. Handle missing values
        4. Scale features
        5. Split into training and testing sets
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        # Calculate technical indicators
        self.calculate_technical_indicators()
        
        # Create target variable (1 if price goes up, 0 if price goes down)
        self.data['Target'] = (self.data['Close'].shift(-1) > self.data['Close']).astype(int)
        
        # Select base features that are most important for prediction
        feature_columns = [
            'Close',          # Current price
            'Returns',        # Daily returns
            'SMA20',         # Short-term trend
            'SMA50',         # Medium-term trend
            'RSI',           # Momentum
            'MACD',          # Trend strength
            'BB_upper',      # Upper Bollinger Band
            'BB_lower',      # Lower Bollinger Band
            'Volume'         # Trading volume
        ]
        
        # Ensure all required columns exist
        print("Available columns:", self.data.columns.tolist())
        
        # Handle NaN values for each feature type
        for col in feature_columns:
            # First try forward fill
            self.data[col] = self.data[col].ffill()
            # Then backward fill any remaining NaNs
            self.data[col] = self.data[col].bfill()
        
        # Drop rows where target is NaN (can't predict these anyway)
        self.data = self.data.dropna(subset=['Target']).copy()
        
        if len(self.data) < 100:  # Require at least 100 data points
            raise ValueError("Insufficient data available after removing NaN values")
        
        # Prepare features and target
        self.X = self.data[feature_columns].copy()
        self.y = self.data['Target'].copy()
        
        print(f"Data shape before preprocessing: {self.X.shape}")
        
        # Replace any infinite values with NaN
        self.X = self.X.replace([np.inf, -np.inf], np.nan)
        
        # Check for columns that are entirely NaN
        nan_columns = self.X.columns[self.X.isna().all()].tolist()
        if nan_columns:
            print(f"Removing columns with all NaN values: {nan_columns}")
            self.X = self.X.drop(columns=nan_columns)
            feature_columns = [col for col in feature_columns if col not in nan_columns]
        
        # For remaining features, handle NaN values
        for column in self.X.columns:
            # Calculate the percentage of NaN values
            nan_percent = self.X[column].isna().mean() * 100
            if nan_percent > 0:
                print(f"Column {column} has {nan_percent:.2f}% NaN values")
            
            # If more than 30% NaN, drop the column
            if nan_percent > 30:
                print(f"Dropping column {column} due to high NaN percentage")
                self.X = self.X.drop(columns=[column])
                feature_columns.remove(column)
                continue
            
            # For remaining NaN values, use forward fill then backward fill
            self.X[column] = self.X[column].fillna(method='ffill')
            self.X[column] = self.X[column].fillna(method='bfill')
            
            # If any NaN still remain, use median
            if self.X[column].isna().any():
                median_val = self.X[column].median()
                self.X[column] = self.X[column].fillna(median_val)
        
        print(f"Data shape after NaN handling: {self.X.shape}")
        
        # Scale the features
        scaled_features = self.scaler.fit_transform(self.X)
        self.X = pd.DataFrame(scaled_features, columns=self.X.columns, index=self.X.index)
        
        # Final verification
        if self.X.isna().any().any():
            # Print which columns still have NaN values
            nan_cols = self.X.columns[self.X.isna().any()].tolist()
            print(f"Columns still containing NaN values: {nan_cols}")
            raise ValueError("NaN values still present after preprocessing")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, shuffle=False
        )
        
        return self.X, self.y
        
    def evaluate_model(self, model, y_pred, model_name):
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        
        print(f"\n{model_name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
        plt.close()
        
        # ROC Curve
        y_pred_proba = getattr(model, "predict_proba", lambda x: np.vstack((1-x, x)).T)(self.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.savefig(f'roc_curve_{model_name.lower().replace(" ", "_")}.png')
        plt.close()
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': roc_auc
        }
    
    def train_all_models(self):
        """
        Train and compare all available models.
        
        Steps:
        1. Train each model with optimized parameters
        2. Compare performance metrics
        3. Select best performing model
        
        Returns:
            tuple: (best_model, model_name, best_accuracy)
        """
        # Train Logistic Regression
        print("\nTraining Logistic Regression...")
        lr_params = {
            'C': [0.001, 0.01, 0.1, 1.0, 10.0],
            'max_iter': [2000],
            'class_weight': ['balanced', None],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }
        lr_model = LogisticRegression(random_state=42)
        lr_grid = GridSearchCV(lr_model, lr_params, cv=5, scoring='f1', n_jobs=-1)
        lr_grid.fit(self.X_train, self.y_train)
        lr_pred = lr_grid.predict(self.X_test)
        lr_metrics = self.evaluate_model(lr_grid.best_estimator_, lr_pred, "Logistic Regression")
        print(f"Best parameters: {lr_grid.best_params_}")
        
        # Train Random Forest
        print("\nTraining Random Forest...")
        rf_params = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5]
        }
        rf_model = RandomForestClassifier(random_state=42)
        rf_grid = GridSearchCV(rf_model, rf_params, cv=5, scoring='accuracy')
        rf_grid.fit(self.X_train, self.y_train)
        rf_pred = rf_grid.predict(self.X_test)
        rf_metrics = self.evaluate_model(rf_grid.best_estimator_, rf_pred, "Random Forest")
        print(f"Best parameters: {rf_grid.best_params_}")
        
        # Train XGBoost
        print("\nTraining XGBoost...")
        xgb_params = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1]
        }
        xgb_model = xgb.XGBClassifier(random_state=42)
        xgb_grid = GridSearchCV(xgb_model, xgb_params, cv=5, scoring='accuracy')
        xgb_grid.fit(self.X_train, self.y_train)
        xgb_pred = xgb_grid.predict(self.X_test)
        xgb_metrics = self.evaluate_model(xgb_grid.best_estimator_, xgb_pred, "XGBoost")
        print(f"Best parameters: {xgb_grid.best_params_}")
        
        # Compare models
        models = {
            'Logistic Regression': (lr_grid.best_estimator_, lr_metrics),
            'Random Forest': (rf_grid.best_estimator_, rf_metrics),
            'XGBoost': (xgb_grid.best_estimator_, xgb_metrics)
        }
        
        # Select best model based on accuracy
        best_model_name = max(models.items(), key=lambda x: x[1][1]['accuracy'])[0]
        best_model = models[best_model_name][0]
        
        print(f"\nBest model: {best_model_name}")
        print(f"Best model accuracy: {models[best_model_name][1]['accuracy']:.4f}")
        
        # Plot model comparison
        self.plot_model_comparison(models)
        
        return best_model
    
    def plot_model_comparison(self, models):
        """Plot comparison of model metrics"""
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        model_names = list(models.keys())
        
        # Prepare data for plotting
        data = []
        for metric in metrics:
            for model_name in model_names:
                data.append({
                    'Model': model_name,
                    'Metric': metric.capitalize(),
                    'Value': models[model_name][1][metric]
                })
        
        df = pd.DataFrame(data)
        
        # Create grouped bar plot
        plt.figure(figsize=(12, 6))
        g = sns.barplot(data=df, x='Metric', y='Value', hue='Model')
        plt.title('Model Performance Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('model_comparison.png')
        plt.close()
        
        # Return the best model
        best_model_name = max(models.items(), key=lambda x: x[1][1]['accuracy'])[0]
        return models[best_model_name][0]

        self.current_model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_pred = self.current_model.predict(self.X_test)
        
        # Evaluate the model
        results = self.evaluate_model(y_pred, "Random Forest")
        print(f"Best parameters: {self.current_model.best_params_}")
        return results
    
    def train_xgboost(self):
        """
        Train and optimize an XGBoost model.
        
        Performs grid search over:
        - Learning rate
        - Maximum depth
        - Number of estimators
        
        Returns:
            tuple: (best_model, accuracy, precision, recall, f1)
        """
        # Define parameter grid for GridSearchCV
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1],
            'n_estimators': [100, 200]
        }
        
        # Create and train model with GridSearchCV
        self.current_model = GridSearchCV(xgb.XGBClassifier(random_state=42), param_grid, cv=5)
        self.current_model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_pred = self.current_model.predict(self.X_test)
        
        # Evaluate the model
        results = self.evaluate_model(y_pred, "XGBoost")
        print(f"Best parameters: {self.current_model.best_params_}")
        return results

def main():
    # Get stock data
    symbol = "AAPL"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3*365)
    
    analyzer = StockAnalyzer(symbol, start_date, end_date)
    analyzer.download_data()
    analyzer.preprocess_data()
    analyzer.add_technical_indicators()
    
    # Initialize predictor
    predictor = StockPredictor(analyzer.data)
    predictor.prepare_data()
    
    # Train and evaluate models
    results = {}
    
    print("\nTraining and evaluating models...")
    
    # Logistic Regression
    results['Logistic Regression'] = predictor.train_logistic_regression()
    
    # Random Forest
    results['Random Forest'] = predictor.train_random_forest()
    
    # XGBoost
    results['XGBoost'] = predictor.train_xgboost()
    
    # Compare models
    print("\nModel Comparison:")
    comparison_df = pd.DataFrame(results).round(4)
    print(comparison_df)
    
    # Plot model comparison
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    plt.figure(figsize=(12, 6))
    comparison_df.plot(kind='bar')
    plt.title('Model Comparison')
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()

def main():
    # Download and analyze stock data
    symbol = 'AAPL'
    start_date = datetime.now() - timedelta(days=365*2)  # 2 years of data
    end_date = datetime.now()
    
    print(f"Downloading data for {symbol}...")
    analyzer = StockAnalyzer(symbol, start_date, end_date)
    analyzer.download_data()
    analyzer.add_technical_indicators()
    
    print("\nPreparing data...")
    predictor = StockPredictor(analyzer.data)
    predictor.prepare_data()
    
    # Train and evaluate all models
    print("\nTraining and evaluating models...")
    best_model = predictor.train_all_models()
    
    # Make predictions for tomorrow
    last_data = predictor.X.iloc[-1:]
    prediction = best_model.predict(last_data)
    probability = best_model.predict_proba(last_data)[0]
    
    print(f"\nPrediction for tomorrow: {'Up' if prediction[0] == 1 else 'Down'}")
    print(f"Probability: {max(probability):.2%}")

if __name__ == "__main__":
    main()
