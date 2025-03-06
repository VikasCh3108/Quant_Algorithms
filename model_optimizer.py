import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import randint, uniform
import optuna
import logging

class ModelOptimizer:
    def __init__(self, n_splits=5, random_state=42):
        """
        Initialize the model optimizer.
        
        Args:
            n_splits (int): Number of splits for time series cross-validation
            random_state (int): Random seed for reproducibility
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.tscv = TimeSeriesSplit(n_splits=n_splits)
        self.setup_logging()
        
    def setup_logging(self):
        """Set up logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('model_optimization.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def create_custom_scorer(self):
        """Create a custom scoring metric that combines accuracy, precision, and Sharpe ratio"""
        return {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score),
            'recall': make_scorer(recall_score),
            'f1': make_scorer(f1_score)
        }

    def optimize_xgboost(self, X, y, method='grid'):
        """
        Optimize XGBoost parameters using either grid search or random search.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            method (str): Optimization method ('grid' or 'random')
        
        Returns:
            dict: Best parameters and scores
        """
        self.logger.info(f"Starting XGBoost optimization using {method} search")
        
        if method == 'grid':
            param_grid = {
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': [100, 200, 300],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            search = GridSearchCV(
                XGBClassifier(random_state=self.random_state),
                param_grid,
                cv=self.tscv,
                scoring=self.create_custom_scorer(),
                refit='f1',
                n_jobs=-1
            )
        else:
            param_dist = {
                'max_depth': randint(3, 10),
                'learning_rate': uniform(0.01, 0.3),
                'n_estimators': randint(50, 500),
                'min_child_weight': randint(1, 7),
                'subsample': uniform(0.7, 0.3),
                'colsample_bytree': uniform(0.7, 0.3)
            }
            search = RandomizedSearchCV(
                XGBClassifier(random_state=self.random_state),
                param_dist,
                n_iter=50,
                cv=self.tscv,
                scoring=self.create_custom_scorer(),
                refit='f1',
                n_jobs=-1,
                random_state=self.random_state
            )

        search.fit(X, y)
        self.logger.info(f"Best XGBoost parameters: {search.best_params_}")
        self.logger.info(f"Best XGBoost score: {search.best_score_}")
        
        return {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'cv_results': search.cv_results_
        }

    def optimize_random_forest(self, X, y, method='grid'):
        """
        Optimize Random Forest parameters using either grid search or random search.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            method (str): Optimization method ('grid' or 'random')
        
        Returns:
            dict: Best parameters and scores
        """
        self.logger.info(f"Starting Random Forest optimization using {method} search")
        
        if method == 'grid':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }
            search = GridSearchCV(
                RandomForestClassifier(random_state=self.random_state),
                param_grid,
                cv=self.tscv,
                scoring=self.create_custom_scorer(),
                refit='f1',
                n_jobs=-1
            )
        else:
            param_dist = {
                'n_estimators': randint(50, 500),
                'max_depth': randint(3, 10),
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 10),
                'max_features': ['sqrt', 'log2']
            }
            search = RandomizedSearchCV(
                RandomForestClassifier(random_state=self.random_state),
                param_dist,
                n_iter=50,
                cv=self.tscv,
                scoring=self.create_custom_scorer(),
                refit='f1',
                n_jobs=-1,
                random_state=self.random_state
            )

        search.fit(X, y)
        self.logger.info(f"Best Random Forest parameters: {search.best_params_}")
        self.logger.info(f"Best Random Forest score: {search.best_score_}")
        
        return {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'cv_results': search.cv_results_
        }

    def optimize_logistic_regression(self, X, y):
        """
        Optimize Logistic Regression parameters using grid search.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
        
        Returns:
            dict: Best parameters and scores
        """
        self.logger.info("Starting Logistic Regression optimization")
        
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'class_weight': [None, 'balanced'],
            'max_iter': [1000]
        }
        
        search = GridSearchCV(
            LogisticRegression(random_state=self.random_state),
            param_grid,
            cv=self.tscv,
            scoring=self.create_custom_scorer(),
            refit='f1',
            n_jobs=-1
        )
        
        search.fit(X, y)
        self.logger.info(f"Best Logistic Regression parameters: {search.best_params_}")
        self.logger.info(f"Best Logistic Regression score: {search.best_score_}")
        
        return {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'cv_results': search.cv_results_
        }

    def optimize_with_optuna(self, X, y, model_type='xgboost', n_trials=100):
        """
        Optimize model parameters using Optuna for advanced hyperparameter optimization.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            model_type (str): Type of model to optimize ('xgboost' or 'random_forest')
            n_trials (int): Number of optimization trials
        
        Returns:
            dict: Best parameters and scores
        """
        self.logger.info(f"Starting Optuna optimization for {model_type}")
        
        def objective(trial):
            if model_type == 'xgboost':
                params = {
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                    'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0)
                }
                model = XGBClassifier(**params, random_state=self.random_state)
            else:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2'])
                }
                model = RandomForestClassifier(**params, random_state=self.random_state)
            
            scores = []
            for train_idx, val_idx in self.tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model.fit(X_train, y_train)
                pred = model.predict(X_val)
                score = f1_score(y_val, pred)
                scores.append(score)
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        self.logger.info(f"Best Optuna parameters: {study.best_params}")
        self.logger.info(f"Best Optuna score: {study.best_value}")
        
        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'study': study
        }

if __name__ == "__main__":
    # Example usage
    from stock_analysis import StockAnalyzer
    from stock_predictor import StockPredictor
    
    # Get stock data
    analyzer = StockAnalyzer('AAPL', '2023-01-01', '2025-03-05')
    data = analyzer.download_data()
    
    # Prepare features and target
    predictor = StockPredictor(data)
    X, y = predictor.prepare_data()
    
    # Initialize optimizer
    optimizer = ModelOptimizer(n_splits=5)
    
    # Optimize models using different methods
    xgb_grid_results = optimizer.optimize_xgboost(X, y, method='grid')
    rf_random_results = optimizer.optimize_random_forest(X, y, method='random')
    lr_results = optimizer.optimize_logistic_regression(X, y)
    
    # Advanced optimization using Optuna
    optuna_results = optimizer.optimize_with_optuna(X, y, model_type='xgboost', n_trials=50)
    
    # Log results
    logging.info("Optimization completed. Check model_optimization.log for detailed results.")
