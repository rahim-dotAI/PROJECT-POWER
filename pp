# Import necessary libraries
import os
import logging
import json
import smtplib
from logging.handlers import RotatingFileHandler
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import backtrader as bt
from scipy.stats import norm

# Machine Learning libraries
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, confusion_matrix, classification_report
)
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed

# XGBoost
from xgboost import XGBClassifier, XGBRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input  # Keras layers are now in tensorflow.keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# Hyperparameter optimization
from hyperopt import hp, tpe, fmin, space_eval

# Importing necessary functions from TA-Lib
from talib import RSI, MACD, EMA, SMA
from talib import CDLENGULFING, CDLDOJI, CDLMORNINGSTAR, CDLEVENINGSTAR, CDLSHOOTINGSTAR, CDLDARKCLOUDCOVER


# Import custom functions for candlestick analysis
from candlestick_functions import CANDLE

# Function to configure logging
def configure_logging(log_file_path):
    """
    Configure logging with a specified log file path.

    Parameters:
        log_file_path (str): Path for the log file.

    Returns:
        None
    """
    try:
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        # Avoid adding duplicate handlers
        if not logger.hasHandlers():
            formatter = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            handler = RotatingFileHandler(log_file_path, maxBytes=1024, backupCount=3)
            handler.setFormatter(formatter)
            logger.addHandler(handler)
    except FileNotFoundError as file_err:
        print(f"Error creating log directory: {file_err}")
    except Exception as e:
        print(f"An error occurred while configuring logging: {str(e)}")

# Example usage of the logging configuration
def example_usage_of_logging(logger):
    """
    Example usage of logging with different log levels.

    Returns:
        None
    """
    try:
        logger.debug('This is a debug message')
        logger.info('This is an info message')
        logger.warning('This is a warning message')
        logger.error('This is an error message')
        logger.critical('This is a critical message')
    except Exception as e:
        print(f"An error occurred while logging: {str(e)}")

# Function to write configuration data to a JSON file
def write_configuration_to_json(config_data, config_file_path):
    """
    Write configuration data to a JSON file.

    Parameters:
        config_data (dict): Configuration data.
        config_file_path (str): Path for the JSON configuration file.

    Returns:
        None
    """
    try:
        os.makedirs(os.path.dirname(config_file_path), exist_ok=True)
        with open(config_file_path, 'w') as config_file:
            json.dump(config_data, config_file, indent=4)
        print(f"Configuration data has been saved to {config_file_path}")
    except FileNotFoundError as file_err:
        print(f"Error writing configuration data - file not found: {file_err}")
    except Exception as e:
        print(f"Error saving configuration data: {str(e)}")

# Function to load configuration data from a JSON file
def load_configuration_from_json(config_file_path):
    """
    Load configuration data from a JSON file.

    Parameters:
        config_file_path (str): Path for the JSON configuration file.

    Returns:
        dict or None: Loaded configuration data if successful, None otherwise.
    """
    try:
        with open(config_file_path, 'r') as config_file:
            config_data = json.load(config_file)
        print("Configuration data loaded successfully.")
        return config_data
    except FileNotFoundError as file_err:
        print(f"Error loading configuration data - file not found: {file_err}")
    except Exception as e:
        print(f"Error loading configuration data: {str(e)}")
        return None

# Main code to set up logging and demonstrate logging and JSON operations
log_file_path = 'app.log'
configure_logging(log_file_path)
logger = logging.getLogger(__name__)

example_usage_of_logging(logger)

# Function to initialize machine learning models
def initialize_ml_models():
    """
    Initialize machine learning models.

    Returns:
        tuple: Initialized RandomForestClassifier, LogisticRegression, and SVC models.
    """
    try:
        rf_model = RandomForestClassifier(random_state=42)
        lr_model = LogisticRegression(random_state=42)
        svm_model = SVC(random_state=42)
        return rf_model, lr_model, svm_model  # Return the initialized models
    except Exception as e:
        print(f"Error initializing machine learning models: {str(e)}")
        return None, None, None

#def download_financial_data(symbols):
    """
    Download financial data using Yahoo Finance.

    Parameters:
        symbols (list): List of stock symbols.

    Returns:
        pd.DataFrame or None: Downloaded financial data if successful, None otherwise.
    """
    start_date = "2010-01-01"  # Start date for data download
    end_date = "2023-09-19"     # End date for data download

    try:
        data = yf.download(symbols, start=start_date, end=end_date)
        return data
    except Exception as e:
        print(f"Error downloading data: {str(e)}")
        return None

# Function to save data to a CSV file
def save_data_to_csv(data, file_path):
    """
    Save data to a CSV file.

    Parameters:
        data (pd.DataFrame): Data to be saved.
        file_path (str): Path for the CSV file.

    Returns:
        None
    """
    try:
        data.to_csv(file_path)
        print("Data saved successfully to:", file_path)
    except Exception as e:
        print(f"Error saving data to CSV: {str(e)}")

# Function to check the availability of historical data file
def check_historical_data_availability(file_path):
    """
    Check the availability of historical data file.

    Parameters:
        file_path (str): Path of the historical data file.

    Returns:
        bool: True if file exists, False otherwise.
    """
    return os.path.exists(file_path)

# Define your configuration data as a Python dictionary
config_data = {
    'log_file_path': 'path',  # Modify this with the desired log file path
    'log_level': 'INFO',
    'log_format': '%(asctime)s [%(levelname)s] - %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S'
}

# Specify the path for the new JSON configuration file
config_file_path = os.path.join('config', 'config.json')  # Modify this with the desired config file path

# Configure logging using the specified log file path
configure_logging(os.path.join(config_data['log_file_path'], 'your_log_file.log'))  # Adjust path

# Example usage of logging
example_usage_of_logging()

# Write configuration data to a JSON file
write_configuration_to_json(config_data, config_file_path)

# Load configuration data from the JSON file
loaded_config_data = load_configuration_from_json(config_file_path)

# Initialize machine learning models
rf_model, lr_model, svm_model = initialize_ml_models()

# Define symbols, start date, and end date for financial data download
symbols = ["USDJPY", "GBPUSD", "AUDUSD", "USDCAD", "NZDUSD", "EURUSD"]
start_date = "2010-01-01"
end_date = "2023-09-19"

# Specify the file path for saving financial data
file_path = os.path.join('path', 'to', 'directory', 'forex_data.csv')  # Modify this with the desired file path

# Check if historical data is available; if not, download and save it
if not check_historical_data_availability(file_path):
    financial_data = download_financial_data(symbols, start_date, end_date)

    if financial_data is not None:
        save_data_to_csv(financial_data, file_path)
else:
    print("Historical data is already available.")
    
# Function for consistent dataset splitting
def split_dataset(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

# Check if features exist in the dataset
if 'features' in data.columns:
    print("Features exist in the dataset.")
else:
    print("Features do not exist in the dataset.")

# Check if the classification target exists in the dataset
if 'Trend_Direction' in data.columns:
    print("Classification target exists in the dataset.")
else:
    print("Classification target does not exist in the dataset.")

# Check if the volatility target exists in the dataset
if 'Volatility' in data.columns:
    print("Volatility target exists in the dataset.")
else:
    print("Volatility target does not exist in the dataset.")

# Check if the order execution target exists in the dataset
if 'Order_Execution_Target' in data.columns:
    print("Order execution target exists in the dataset.")
else:
    print("Order execution target does not exist in the dataset.")

# Split the dataset for classification
X_train_class, X_test_class, y_train_class, y_test_class = split_dataset(X, y_classification)

# Split the dataset for volatility prediction
X_train_volatility, X_test_volatility, y_train_volatility, y_test_volatility = split_dataset(X, y_volatility)

# Split the dataset for order execution (if needed)
X_train_order_execution, X_test_order_execution, y_train_order_execution, y_test_order_execution = split_dataset(X, y_order_execution)

# Define classification models
classification_models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'Support Vector Machine': SVC(random_state=42),
}

# Define volatility prediction models
volatility_models = {
    'Linear Regression': LinearRegression(),
    'Random Forest Regressor': RandomForestRegressor(random_state=42),
    'Support Vector Regressor': SVR(),
    'XGBoost Regressor': XGBRegressor(objective='reg:squarederror', random_state=42),
}

# Define order execution models
order_execution_models = {
    'Random Forest Classifier': RandomForestClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'Support Vector Machine': SVC(random_state=42),
    'XGBoost Classifier': XGBClassifier(objective='binary:logistic', random_state=42),
}

# Print statements for clarity
print("Dataset Splitting and Model Definition Complete.")
print(f"Number of features: {X.shape[1]}")
print(f"Number of samples: {X.shape[0]}")
print("Classification Models:")
print(classification_models)
print("Volatility Prediction Models:")
print(volatility_models)
print("Order Execution Models:")
print(order_execution_models)



def perform_cross_validation(models, X_train, y_train, num_folds, scoring_metrics):
    cv_results = {}

    for model_name, model in models.items():
        if 'order_execution' in model_name:
            cv_scores = cross_val_score(model, X_train, y_train, cv=num_folds, scoring=scoring_metrics)
        else:
            cv_scores = cross_val_score(model, X_train, y_train, cv=num_folds, scoring='neg_mean_squared_error')
        
        if 'order_execution' in model_name:
            cv_results[model_name] = {metric: np.mean(cv_scores[i]) for i, metric in enumerate(scoring_metrics)}
        else:
            cv_results[model_name] = {'neg_mean_squared_error': -np.mean(cv_scores)}

    return cv_results

# Number of cross-validation folds (e.g., 5-fold cross-validation)
num_folds = 7

# Define evaluation metrics (e.g., accuracy, precision, recall, F1-score, ROC AUC)
scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'roc_auc_ovo']

# Perform cross-validation for classification models
classification_cv_results = perform_cross_validation(classification_models, X_train_class, y_train_class, num_folds, scoring_metrics)

# Perform cross-validation for volatility prediction models
volatility_cv_results = perform_cross_validation(volatility_models, X_train_volatility, y_train_volatility, num_folds, scoring_metrics)

# Perform cross-validation for order execution models (if needed)
order_execution_cv_results = perform_cross_validation(order_execution_models, X_train_order_execution, y_train_order_execution, num_folds, scoring_metrics)

# Print cross-validation results
print("Classification Cross-Validation Results:")
for model_name, results in classification_cv_results.items():
    print(model_name)
    for metric, score in results.items():
        print(f"{metric}: {score}")
    print()

print("Volatility Prediction Cross-Validation Results:")
for model_name, results in volatility_cv_results.items():
    print(model_name)
    for metric, score in results.items():
        print(f"{metric}: {score}")
    print()

if order_execution_cv_results:
    print("Order Execution Cross-Validation Results:")
    for model_name, results in order_execution_cv_results.items():
        print(model_name)
        for metric, score in results.items():
            print(f"{metric}: {score}")
        print()



def evaluate_classification_models(models, X_train, y_train, X_test, y_test, metrics):
    results = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[model_name] = {}
        for metric_name, metric_func in metrics.items():
            score = metric_func(y_test, y_pred)
            results[model_name][metric_name] = score
        results[model_name]['Confusion Matrix'] = confusion_matrix(y_test, y_pred)
        results[model_name]['Classification Report'] = classification_report(y_test, y_pred)
    return results

def evaluate_volatility_models(models, X_train, y_train, X_test, y_test, metrics):
    results = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[model_name] = {}
        for metric_name, metric_func in metrics.items():
            score = metric_func(y_test, y_pred)
            results[model_name][metric_name] = score
    return results

def evaluate_order_execution_models(models, X_train, y_train, X_test, y_test, metrics):
    results = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[model_name] = {}
        for metric_name, metric_func in metrics.items():
            score = metric_func(y_test, y_pred)
            results[model_name][metric_name] = score
        results[model_name]['Confusion Matrix'] = confusion_matrix(y_test, y_pred)
        results[model_name]['Classification Report'] = classification_report(y_test, y_pred)
    return results

# Train and evaluate classification models on the test set
classification_test_results = evaluate_classification_models(classification_models, X_train_class, y_train_class, X_test_class, y_test_class, classification_metrics)

# Train and evaluate volatility prediction models on the test set
volatility_test_results = evaluate_volatility_models(volatility_models, X_train_volatility, y_train_volatility, X_test_volatility, y_test_volatility, volatility_metrics)

# Train and evaluate order execution models on the test set (if needed)
if order_execution_models:
    order_execution_test_results = evaluate_order_execution_models(order_execution_models, X_train_order_execution, y_train_order_execution, X_test_order_execution, y_test_order_execution, order_execution_metrics)

# Print test set results for classification models
print("Classification Test Set Results:")
for model_name, results in classification_test_results.items():
    print(model_name)
    for metric, score in results.items():
        if metric == 'Confusion Matrix' or metric == 'Classification Report':
            print(f"{metric}:\n{score}")
        else:
            print(f"{metric}: {score}")
    print()


# Function to handle evaluation of a single model
def evaluate_model(model, X_test, y_test):
    accuracy = model.score(X_test, y_test)
    predictions = model.predict(X_test)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    
    return {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}

# Function to print test set results for volatility prediction models
def print_volatility_results(volatility_test_results):
    print("Volatility Prediction Test Set Results:")
    for model_name, results in volatility_test_results.items():
        print(model_name)
        for metric, score in results.items():
            print(f"{metric}: {score}")
        print()

# Function to print test set results for order execution models (if needed)
def print_order_execution_results(order_execution_test_results):
    if order_execution_test_results:
        print("Order Execution Test Set Results:")
        for model_name, results in order_execution_test_results.items():
            print(model_name)
            for metric, score in results.items():
                if metric == 'Confusion Matrix' or metric == 'Classification Report':
                    print(f"{metric}:\n{score}")
                else:
                    print(f"{metric}: {score}")
            print()

# Evaluate models on the test set
rf_results = evaluate_model(rf_model, X_test, y_test)
lr_results = evaluate_model(lr_model, X_test, y_test)
svm_results = evaluate_model(svm_model, X_test, y_test)

# Print test results for each model
volatility_test_results = {
    'Random Forest': rf_results,
    'Logistic Regression': lr_results,
    'SVM': svm_results
}

print_volatility_results(volatility_test_results)



# Extract configuration constants with default values
FOREX_SYMBOLS = config.get("FOREX_SYMBOLS", [])
START_DATE = config.get("START_DATE", "2010-01-01")
END_DATE = config.get("END_DATE", "2023-09-19")
SHORT_EMA_PERIOD = config.get("SHORT_EMA_PERIOD", 10)
LONG_EMA_PERIOD = config.get("LONG_EMA_PERIOD", 20)
RSI_OVERSOLD_THRESHOLD = config.get("RSI_OVERSOLD_THRESHOLD", 30)
RSI_OVERBOUGHT_THRESHOLD = config.get("RSI_OVERBOUGHT_THRESHOLD", 70)
MACD_SHORT_WINDOW = config.get("MACD_SHORT_WINDOW", 12)
MACD_LONG_WINDOW = config.get("MACD_LONG_WINDOW", 26)
EMAIL_NOTIFICATIONS = config.get("EMAIL_NOTIFICATIONS", True)
EMAIL_CONFIG = config.get("EMAIL_CONFIG", {})
ORDER_TYPE = config.get("ORDER_TYPE", "market")
MAX_RISK_PERCENT = config.get("MAX_RISK_PERCENT", 1)
KELLY_FRACTION = config.get("KELLY_FRACTION", 0.5)
TRAINING_PERIOD = config.get("TRAINING_PERIOD", 1000)
TEST_PERIOD = config.get("TEST_PERIOD", 100)
MIN_DATA_POINTS = config.get("MIN_DATA_POINTS", 1000)
REGIME_WINDOW = config.get("REGIME_WINDOW", 100)
CONFIDENCE_LEVEL = config.get("CONFIDENCE_LEVEL", 0.95)
LOG_FILE = config.get("LOG_FILE", "backtest.log")

# Maximum Risk Limit
MAX_RISK_PERCENT = 2 / 100  # 2 percent

# Kelly Criterion Constants
KELLY_FRACTION = 0.5

# Walk-Forward Validation Constants
TRAINING_PERIOD = 252
TEST_PERIOD = 63
MIN_DATA_POINTS = TRAINING_PERIOD + TEST_PERIOD

# Market Regime Detection Constants
REGIME_WINDOW = 50

# VaR Constants
CONFIDENCE_LEVEL = 0.95  # 95% confidence level

# Initialize slippage statistics dictionary

def configure_logging(log_file_path):
    try:
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        logging.config.dictConfig({
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'standard': {
                    'format': '%(asctime)s [%(levelname)s] - %(message)s',
                    'datefmt': '%Y-%m-%d %H:%M:%S'
                },
            },
            'handlers': {
                'file': {
                    'level': 'INFO',
                    'class': 'logging.handlers.RotatingFileHandler',
                    'filename': log_file_path,
                    'maxBytes': 1024,
                    'backupCount': 3,
                    'formatter': 'standard',
                },
            },
            'root': {
                'level': 'INFO',
                'handlers': ['file'],
            },
        })
    except Exception as e:
        logging.error(f"An error occurred while configuring logging: {str(e)}")

if __name__ == '__main__':
    log_file_path = '/path/to/your_log_file.log'
    configure_logging(log_file_path)

# Modularized Email Alert Function
def send_alert(subject: str, message: str):
    try:
        if EMAIL_NOTIFICATIONS:
            sender_email = EMAIL_CONFIG.get("sender_email")
            receiver_email = EMAIL_CONFIG.get("receiver_email")
            password = os.getenv("EMAIL_PASSWORD")  # Use environment variable for security

            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = receiver_email
            msg['Subject'] = subject

            msg.attach(MIMEText(message, 'plain'))

            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(sender_email, password)
            text = msg.as_string()
            server.sendmail(sender_email, receiver_email, text)
            server.quit()
            logging.info(f"Alert email sent: {subject}")
    except Exception as e:
        error_msg = f"Error sending alert email: {str(e)}"
        logging.error(error_msg)

def handle_error(msg):
    logging.error(f"Error: {msg}")
    raise ValueError(msg)

def calculate_ema(data: pd.DataFrame, period: int) -> pd.DataFrame:
    if not isinstance(period, int) or period <= 0:
        handle_error("Period must be a positive integer.")

    try:
        alpha = 2 / (period + 1)
        data['EMA'] = data['Close'].ewm(span=period, adjust=False).mean()
        return data
    except Exception as e:
        handle_error(f"EMA calculation: {str(e)}")

class SMACalculationError(Exception):
    pass

def calculate_sma(data, period):
    if not isinstance(data, pd.DataFrame):
        handle_error("Input 'data' should be a pandas DataFrame.")
    
    if not isinstance(period, int) or period <= 0:
        handle_error("Period should be a positive integer greater than zero.")
    
    if 'Close' not in data.columns:
        handle_error("Input data must contain a 'Close' column.")

    try:
        sma_values = data['Close'].rolling(window=period).mean()
        data = data.assign(SMA=sma_values)
        return data
    except Exception as e:
        handle_error(f"SMA calculation: {str(e)}")

def calculate_rsi(data, period=14):
    try:
        if not isinstance(data, pd.DataFrame):
            handle_error("Input 'data' should be a pandas DataFrame.")

        if not isinstance(period, int) or period <= 0:
            handle_error("Period should be a positive integer greater than zero.")

        if 'Close' not in data.columns:
            handle_error("Input data must contain a 'Close' column.")

        delta = data['Close'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        data['RSI'] = rsi
        return data.dropna()
    except (ValueError, KeyError) as e:
        handle_error(f"RSI calculation: {str(e)}")


class MACDCalculator:
    """
    Class for calculating Moving Average Convergence Divergence (MACD) and its signal line.
    """

    def __init__(self, data: pd.DataFrame, short_window: int, long_window: int, signal_line_period: int = 9):
        """
        Initialize MACDCalculator instance.

        Parameters:
        - data (pd.DataFrame): Input data with 'Close' column.
        - short_window (int): Short-term moving average window.
        - long_window (int): Long-term moving average window.
        - signal_line_period (int, optional): Period for the signal line (default is 9).
        """
        self.data = data
        self.short_window = short_window
        self.long_window = long_window
        self.signal_line_period = signal_line_period
        self.validate_input()

    def validate_input(self):
        """
        Validate input data and parameters.
        """
        if not isinstance(self.data, pd.DataFrame):
            raise ValueError("Input 'data' should be a pandas DataFrame.")

        if not all(isinstance(window, int) and window > 0 for window in [self.short_window, self.long_window, self.signal_line_period]):
            raise ValueError("Short and long windows, and signal line period should be positive integers greater than zero.")

        if 'Close' not in self.data.columns:
            raise ValueError("Input data must contain a 'Close' column.")

        if self.data['Close'].isnull().any():
            raise ValueError("Input data contains missing (NaN) values in the 'Close' column.")

    def calculate_ema(self, column_name, span):
        """
        Calculate Exponential Moving Average (EMA) for the specified column.

        Parameters:
        - column_name (str): Name of the column for EMA calculation.
        - span (int): Span parameter for EMA.

        Returns:
        - pd.Series: EMA values.
        """
        return self.data[column_name].ewm(span=span, adjust=False).mean()

    def calculate_macd(self):
        """
        Calculate MACD values and update the 'MACD' column in the data.
        """
        short_ema = self.calculate_ema('Close', self.short_window)
        long_ema = self.calculate_ema('Close', self.long_window)
        self.data['MACD'] = short_ema - long_ema
        return self

    def calculate_signal_line(self):
        """
        Calculate MACD Signal Line values and update the 'Signal_Line' column in the data.
        """
        self.data['Signal_Line'] = self.calculate_ema('MACD', self.signal_line_period)
        return self

    def calculate_macd_and_signal(self):
        """
        Calculate both MACD and Signal Line values.
        """
        self.calculate_macd()
        self.calculate_signal_line()
        return self.data

def configure_logging(log_filename, log_level=logging.INFO):
    """
    Configure logging settings.

    Parameters:
    - log_filename (str): Name of the log file.
    - log_level (int, optional): Logging level (default is INFO).
    """
    logging.basicConfig(filename=log_filename, level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def main():
    """
    Main function to demonstrate the usage of MACDCalculator.
    """
    data = pd.DataFrame({'Close': [100, 105, 110, 115, 120, 125, 130, 135, 140, 145]})
    short_window = 3
    long_window = 6
    signal_line_period = 9

    try:
        configure_logging('macd_log.txt', logging.INFO)
        macd_calculator = MACDCalculator(data, short_window, long_window, signal_line_period)
        result = macd_calculator.calculate_macd_and_signal()
        print(result)
    except ValueError as e:
        error_msg = f"Error calculating MACD: {str(e)}"
        logging.error(error_msg)
        raise ValueError(error_msg)

if __name__ == "__main__":
    main()


def calculate_candlestick_patterns(data):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input 'data' should be a pandas DataFrame.")

    if not all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
        raise ValueError("Input data must contain 'Open', 'High', 'Low', and 'Close' columns.")

    patterns = {
        'Bullish Engulfing': bullish_engulfing,
        'Bearish Engulfing': bearish_engulfing,
        'Doji': doji,
        'Morning Star': morning_star,
        'Evening Star': evening_star,
        'Shooting Star': shooting_star,
        'Dark Cloud Cover': dark_cloud_cover
        # Add more patterns and corresponding functions here
    }

    initialize_pattern_columns(data, patterns)

    calculate_and_update_patterns(data, patterns)

    return data

def initialize_pattern_columns(data, patterns):
    for pattern in patterns:
        data[pattern] = False
        data[f'{pattern} Confidence'] = 0.0

def calculate_and_update_patterns(data, patterns):
    for i in range(2, len(data)):
        for pattern, pattern_func in patterns.items():
            detected, confidence = pattern_func(data, i)
            data.at[i, pattern] = detected
            data.at[i, f'{pattern} Confidence'] = confidence

def bullish_engulfing(data, i):
    if (
        data['Close'].iloc[i] > data['Open'].iloc[i] and
        data['Open'].iloc[i - 1] > data['Close'].iloc[i - 1] and
        data['Close'].iloc[i - 1] > data['Open'].iloc[i] and
        data['Open'].iloc[i] < data['Close'].iloc[i - 1]
    ):
        confidence = 0.8
        return True, confidence
    else:
        return False, 0.0

def bearish_engulfing(data, i):
    if (
        data['Open'].iloc[i] > data['Close'].iloc[i] and
        data['Close'].iloc[i - 1] > data['Open'].iloc[i - 1] and
        data['Open'].iloc[i - 1] > data['Close'].iloc[i] and
        data['Close'].iloc[i] < data['Open'].iloc[i - 1]
    ):
        confidence = 0.7
        return True, confidence
    else:
        return False, 0.0


def doji(data, i):
    """
    Detects if a candle is a Doji pattern.

    Parameters:
    - data (pd.DataFrame): DataFrame containing financial data.
    - i (int): Index of the current candle.

    Returns:
    - tuple (bool, float): True if Doji pattern is detected, along with confidence score.
    """
    # Validate input parameters
    if not isinstance(data, pd.DataFrame) or not isinstance(i, int) or i < 2 or i >= len(data):
        raise ValueError("Invalid input parameters.")

    # Define a threshold for the body of the candle (a small value)
    body_threshold = 0.01 * (data['High'].iloc[i] - data['Low'].iloc[i])

    # Check if the current candle is a Doji
    if abs(data['Close'].iloc[i] - data['Open'].iloc[i]) < body_threshold:
        # Assign a confidence score for the detected pattern
        confidence = 0.6
        return True, confidence
    else:
        return False, 0.0


def morning_star(data, i):
    """
    Detects if a candle is a Morning Star pattern.

    Parameters:
    - data (pd.DataFrame): DataFrame containing financial data.
    - i (int): Index of the current candle.

    Returns:
    - tuple (bool, float): True if Morning Star pattern is detected, along with confidence score.
    """
    # Validate input parameters
    if not isinstance(data, pd.DataFrame) or not isinstance(i, int) or i < 3 or i >= len(data):
        raise ValueError("Invalid input parameters.")

    # Check if the current candle is a Morning Star pattern
    if (
        # Check the first candle
        data['Close'].iloc[i - 2] < data['Open'].iloc[i - 2] and
        data['Close'].iloc[i - 2] < data['Close'].iloc[i - 1] and

        # Check the second candle (the Doji or small body)
        data['Open'].iloc[i - 1] > data['Close'].iloc[i - 2] and
        data['Close'].iloc[i - 1] < data['Open'].iloc[i] and

        # Check the third candle
        data['Open'].iloc[i] < data['Close'].iloc[i - 2] and
        data['Close'].iloc[i] > data['Close'].iloc[i - 1]
    ):
        # Assign a confidence score for the detected pattern
        confidence = 0.9
        return True, confidence
    else:
        return False, 0.0


def evening_star(data, i):
    """
    Detects if a candle is an Evening Star pattern.

    Parameters:
    - data (pd.DataFrame): DataFrame containing financial data.
    - i (int): Index of the current candle.

    Returns:
    - tuple (bool, float): True if Evening Star pattern is detected, along with confidence score.
    """
    # Validate input parameters
    if not isinstance(data, pd.DataFrame) or not isinstance(i, int) or i < 3 or i >= len(data):
        raise ValueError("Invalid input parameters.")

    # Check if the current candle is an Evening Star pattern
    if (
        # Check the first candle
        data['Close'].iloc[i - 2] > data['Open'].iloc[i - 2] and
        data['Close'].iloc[i - 2] > data['Close'].iloc[i - 1] and

        # Check the second candle (the Doji or small body)
        data['Open'].iloc[i - 1] < data['Close'].iloc[i - 2] and
        data['Close'].iloc[i - 1] > data['Open'].iloc[i] and

        # Check the third candle
        data['Open'].iloc[i] > data['Close'].iloc[i - 2] and
        data['Close'].iloc[i] < data['Close'].iloc[i - 1]
    ):
        # Assign a confidence score for the detected pattern
        confidence = 0.7
        return True, confidence
    else:
        return False, 0.0


def shooting_star(data, i):
    """
    Detects if a candle is a Shooting Star pattern.

    Parameters:
    - data (pd.DataFrame): DataFrame containing financial data.
    - i (int): Index of the current candle.

    Returns:
    - tuple (bool, float): True if Shooting Star pattern is detected, along with confidence score.
    """
    # Validate input parameters
    if not isinstance(data, pd.DataFrame) or not isinstance(i, int) or i < 1 or i >= len(data):
        raise ValueError("Invalid input parameters.")

    # Check if the current candle is a Shooting Star pattern
    if (
        data['Open'].iloc[i] < data['Close'].iloc[i] and
        data['Close'].iloc[i] < data['Open'].iloc[i - 1] and
        (data['High'].iloc[i] - data['Close'].iloc[i]) > 2 * (data['Open'].iloc[i] - data['Low'].iloc[i])
    ):
        # Assign a confidence score for the detected pattern
        confidence = 0.6
        return True, confidence
    else:
        return False, 0.0

def dark_cloud_cover(data, i):
    if (
        data['Open'].iloc[i] < data['Close'].iloc[i - 1] and
        data['Close'].iloc[i] > (data['Close'].iloc[i - 1] + data['Open'].iloc[i - 1]) / 2 and
        data['Close'].iloc[i] < data['Open'].iloc[i - 1]
    ):
        confidence = 0.8
        return True, confidence
    else:
        return False, 0.0

def calculate_candlestick_patterns(data):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input 'data' should be a pandas DataFrame.")

    if not all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
        raise ValueError("Input data must contain 'Open', 'High', 'Low', and 'Close' columns.")

    patterns = {
        'Bullish Engulfing': bullish_engulfing,
        'Bearish Engulfing': bearish_engulfing,
        'Doji': doji,
        'Morning Star': morning_star,
        'Evening Star': evening_star,
        'Shooting Star': shooting_star,
        'Dark Cloud Cover': dark_cloud_cover
        # Add more patterns and corresponding functions here
    }

    initialize_pattern_columns(data, patterns)

    calculate_and_update_patterns(data, patterns)

    return data

def initialize_pattern_columns(data, patterns):
    for pattern in patterns:
        data[pattern] = False
        data[f'{pattern} Confidence'] = 0.0

def calculate_and_update_patterns(data, patterns):
    for i in range(2, len(data)):
        for pattern, pattern_func in patterns.items():
            detected, confidence = pattern_func(data, i)
            data.at[i, pattern] = detected
            data.at[i, f'{pattern} Confidence'] = confidence

def split_data(data, validation_size=0.2, random_state=42):
    features = ['Open', 'High', 'Low', 'Close', 'SMA', 'EMA', 'RSI', 'MACD']
    target = 'Trend_Direction'

    X = data[features]
    y = data[target]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_size, random_state=random_state)

    return X_train, X_val, y_train, y_val

# Example usage:
data = pd.DataFrame({
    'Open': [100, 110, 120, 130, 120, 110, 110],
    'High': [115, 125, 135, 140, 130, 125, 125],
    'Low': [95, 105, 115, 125, 115, 105, 105],
    'Close': [110, 120, 130, 125, 115, 115, 120]
})

data_with_patterns = calculate_candlestick_patterns(data)
print(data_with_patterns)

X_train, X_val, y_train, y_val = split_data(data)


# Define classification models
classification_models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'Support Vector Machine': SVC(random_state=42),
}

# Function to train and evaluate classification models
def train_and_evaluate_classification_models(X_train, y_train, X_val, y_val, models):
    """
    Train and evaluate classification models on the validation set.

    Parameters:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation target.
        models (dict): Dictionary of classification models to train and evaluate.

    Returns:
        dict: Dictionary of model evaluation metrics.
    """
    results = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        # Customize the evaluation metrics based on your needs
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='weighted')
        recall = recall_score(y_val, y_pred, average='weighted')
        f1 = f1_score(y_val, y_pred, average='weighted')

        results[model_name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
        }

    return results

# Train and evaluate classification models
classification_results = train_and_evaluate_classification_models(X_train, y_train, X_val, y_val, classification_models)

# Train the classifier
# Initialize a Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)

# Train the Random Forest Classifier on the entire dataset
rf_model.fit(X_train, y_train)

# Predict the target labels on the validation set
y_pred = rf_model.predict(X_val)


# Function to create and evaluate the model
def train_and_evaluate_model(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    return accuracy

# Define the objective function to minimize
def objective(hyperparameters):
    # Create a model using the hyperparameters
    model = create_model(model_name, hyperparameters)

    # Train and evaluate the model on the training and validation data
    score = train_and_evaluate_model(model, X_train, y_train, X_val, y_val)

    # Return the score to minimize (negative of accuracy, for example)
    return -score

# Specify the model name (e.g., 'Random Forest', 'Logistic Regression')
model_name = 'Random Forest'

# Define the hyperparameter search space for the chosen model
# (You may have different hyperparameters for each model)
hyperparameter_space = {
    'n_estimators': hp.quniform('n_estimators', 10, 100, 1),
    'max_depth': hp.quniform('max_depth', 1, 20, 1),
    'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 1),
    'max_features': hp.uniform('max_features', 0.1, 1.0),
}

# Use the Tree of Parzen Estimators (TPE) algorithm for optimization
print("Optimizing Hyperparameters:")
best_hyperparameters = fmin(fn=objective, space=hyperparameter_space, algo=tpe.suggest, max_evals=100)

# Print the best hyperparameters found by Bayesian optimization
print("\nBest hyperparameters:", best_hyperparameters)


# Function to create a machine learning model based on the model_name and hyperparameters
def create_model(model_name, hyperparameters):
    """
    Create a machine learning model.

    Parameters:
    - model_name (str): Name of the model ('Random Forest', 'Logistic Regression', 'Support Vector Machine').
    - hyperparameters (dict): Hyperparameters for the specified model.

    Returns:
    - model: Machine learning model instance.
    """
    try:
        if model_name == 'Random Forest':
            model = RandomForestClassifier(**hyperparameters)
        elif model_name == 'Logistic Regression':
            model = LogisticRegression(**hyperparameters)
        elif model_name == 'Support Vector Machine':
            model = SVC(**hyperparameters)
        # Add more model creation logic as needed
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        
        return model
    except Exception as e:
        print(f"Error creating model: {str(e)}")

# Rest of the code remains unchanged...

# Replace these with your actual data loading logic
X_train, y_train, X_val, y_val, X_test, y_test = load_your_data() 

# Example of an evaluation function for classification tasks (customize as needed)
def evaluate_classification_model(model, X_val, y_val):
    """
    Evaluate a classification model.

    Parameters:
    - model: Trained classification model.
    - X_val: Validation input data.
    - y_val: True labels for validation data.

    Returns:
    - dict: Dictionary containing evaluation metrics.
    """
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    return {
        'Accuracy': accuracy,
        'F1-Score': f1
    }

# Example of an evaluation function for regression tasks (customize as needed)
def evaluate_regression_model(model, X_val, y_val):
    """
    Evaluate a regression model.

    Parameters:
    - model: Trained regression model.
    - X_val: Validation input data.
    - y_val: True labels for validation data.

    Returns:
    - dict: Dictionary containing evaluation metrics.
    """
    y_pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    r2 = r2_score(y_val, y_pred)
    return {
        'RMSE': rmse,
        'R-squared': r2
    }


# Define the model name and hyperparameters
model_name = 'Random Forest'
hyperparameters = {
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 2
}

# Create the model
model = create_model(model_name, hyperparameters)

# Train and evaluate the model on the validation set
y_pred = train_and_evaluate_model(model, X_train, y_train, X_val, y_val)  # Ensure validation set is used during training

# Evaluate the model performance on the validation set
evaluation_metric = evaluate_model_performance(model, X_val, y_val, task_type='classification')

# Print the evaluation result
print(evaluation_metric)

# Initialize a Trials object to track the optimization process
trials = Trials()

# Perform Bayesian optimization to find the best hyperparameters
best = fmin(fn=objective,  # Objective function to minimize
            space=hyperparameter_space,  # Hyperparameter search space
            algo=tpe.suggest,  # Optimization algorithm (Tree-structured Parzen Estimator)
            max_evals=100,  # Maximum number of evaluations
            trials=trials,  # Optimization trials object
            verbose=1)  # Verbosity level (0: silent, 1: progress bar, 2: detailed)

# Retrieve the best hyperparameters and corresponding model name
best_hyperparameters = space_eval(hyperparameter_space, best)
best_model_name = best_hyperparameters['model']
best_hyperparameters = best_hyperparameters['hyperparameters']

# Create the best model with the optimal hyperparameters
best_model = create_model(best_model_name, best_hyperparameters)

# Train and evaluate the best model on the test set
y_pred = train_and_evaluate_model(best_model, X_train, y_train, X_test, y_test)

# Print the best hyperparameters and test accuracy
print("Best Model:", best_model_name)
print("Best Hyperparameters:", best_hyperparameters)
evaluation_metric = evaluate_model_performance(best_model, X_test, y_test, task_type='classification')
print("Test Accuracy:", evaluation_metric['Accuracy'])




# Function for consistent dataset splitting
def split_dataset(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

# Check if features exist in the dataset
if 'features' in data.columns:
    print("Features exist in the dataset.")
else:
    print("Features do not exist in the dataset.")

# Check if the classification target exists in the dataset
if 'Trend_Direction' in data.columns:
    print("Classification target exists in the dataset.")
else:
    print("Classification target does not exist in the dataset.")

# Check if the volatility target exists in the dataset
if 'Volatility' in data.columns:
    print("Volatility target exists in the dataset.")
else:
    print("Volatility target does not exist in the dataset.")

# Check if the order execution target exists in the dataset
if 'Order_Execution_Target' in data.columns:
    print("Order execution target exists in the dataset.")
else:
    print("Order execution target does not exist in the dataset.")

# Split the dataset for classification
X_train_class, X_test_class, y_train_class, y_test_class = split_dataset(X, y_classification)

# Split the dataset for volatility prediction
X_train_volatility, X_test_volatility, y_train_volatility, y_test_volatility = split_dataset(X, y_volatility)

# Split the dataset for order execution (if needed)
X_train_order_execution, X_test_order_execution, y_train_order_execution, y_test_order_execution = split_dataset(X, y_order_execution)

# Define classification models
classification_models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'Support Vector Machine': SVC(random_state=42),
}

# Define volatility prediction models
volatility_models = {
    'Linear Regression': LinearRegression(),
    'Random Forest Regressor': RandomForestRegressor(random_state=42),
    'Support Vector Regressor': SVR(),
    'XGBoost Regressor': XGBRegressor(objective='reg:squarederror', random_state=42),
}

# Define order execution models
order_execution_models = {
    'Random Forest Classifier': RandomForestClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'Support Vector Machine': SVC(random_state=42),
    'XGBoost Classifier': XGBClassifier(objective='binary:logistic', random_state=42),
}

# Print statements for clarity
print("Dataset Splitting and Model Definition Complete.")
print(f"Number of features: {X.shape[1]}")
print(f"Number of samples: {X.shape[0]}")
print("Classification Models:")
print(classification_models)
print("Volatility Prediction Models:")
print(volatility_models)
print("Order Execution Models:")
print(order_execution_models)



def perform_cross_validation(models, X_train, y_train, num_folds, scoring_metrics):
    cv_results = {}

    for model_name, model in models.items():
        if 'order_execution' in model_name:
            cv_scores = cross_val_score(model, X_train, y_train, cv=num_folds, scoring=scoring_metrics)
        else:
            cv_scores = cross_val_score(model, X_train, y_train, cv=num_folds, scoring='neg_mean_squared_error')
        
        if 'order_execution' in model_name:
            cv_results[model_name] = {metric: np.mean(cv_scores[i]) for i, metric in enumerate(scoring_metrics)}
        else:
            cv_results[model_name] = {'neg_mean_squared_error': -np.mean(cv_scores)}

    return cv_results

# Number of cross-validation folds (e.g., 5-fold cross-validation)
num_folds = 7

# Define evaluation metrics (e.g., accuracy, precision, recall, F1-score, ROC AUC)
scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'roc_auc_ovo']

# Perform cross-validation for classification models
classification_cv_results = perform_cross_validation(classification_models, X_train_class, y_train_class, num_folds, scoring_metrics)

# Perform cross-validation for volatility prediction models
volatility_cv_results = perform_cross_validation(volatility_models, X_train_volatility, y_train_volatility, num_folds, scoring_metrics)

# Perform cross-validation for order execution models (if needed)
order_execution_cv_results = perform_cross_validation(order_execution_models, X_train_order_execution, y_train_order_execution, num_folds, scoring_metrics)

# Print cross-validation results
print("Classification Cross-Validation Results:")
for model_name, results in classification_cv_results.items():
    print(model_name)
    for metric, score in results.items():
        print(f"{metric}: {score}")
    print()

print("Volatility Prediction Cross-Validation Results:")
for model_name, results in volatility_cv_results.items():
    print(model_name)
    for metric, score in results.items():
        print(f"{metric}: {score}")
    print()

if order_execution_cv_results:
    print("Order Execution Cross-Validation Results:")
    for model_name, results in order_execution_cv_results.items():
        print(model_name)
        for metric, score in results.items():
            print(f"{metric}: {score}")
        print()



def evaluate_classification_models(models, X_train, y_train, X_test, y_test, metrics):
    results = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[model_name] = {}
        for metric_name, metric_func in metrics.items():
            score = metric_func(y_test, y_pred)
            results[model_name][metric_name] = score
        results[model_name]['Confusion Matrix'] = confusion_matrix(y_test, y_pred)
        results[model_name]['Classification Report'] = classification_report(y_test, y_pred)
    return results

def evaluate_volatility_models(models, X_train, y_train, X_test, y_test, metrics):
    results = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[model_name] = {}
        for metric_name, metric_func in metrics.items():
            score = metric_func(y_test, y_pred)
            results[model_name][metric_name] = score
    return results

def evaluate_order_execution_models(models, X_train, y_train, X_test, y_test, metrics):
    results = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[model_name] = {}
        for metric_name, metric_func in metrics.items():
            score = metric_func(y_test, y_pred)
            results[model_name][metric_name] = score
        results[model_name]['Confusion Matrix'] = confusion_matrix(y_test, y_pred)
        results[model_name]['Classification Report'] = classification_report(y_test, y_pred)
    return results

# Train and evaluate classification models on the test set
classification_test_results = evaluate_classification_models(classification_models, X_train_class, y_train_class, X_test_class, y_test_class, classification_metrics)

# Train and evaluate volatility prediction models on the test set
volatility_test_results = evaluate_volatility_models(volatility_models, X_train_volatility, y_train_volatility, X_test_volatility, y_test_volatility, volatility_metrics)

# Train and evaluate order execution models on the test set (if needed)
if order_execution_models:
    order_execution_test_results = evaluate_order_execution_models(order_execution_models, X_train_order_execution, y_train_order_execution, X_test_order_execution, y_test_order_execution, order_execution_metrics)

# Print test set results for classification models
print("Classification Test Set Results:")
for model_name, results in classification_test_results.items():
    print(model_name)
    for metric, score in results.items():
        if metric == 'Confusion Matrix' or metric == 'Classification Report':
            print(f"{metric}:\n{score}")
        else:
            print(f"{metric}: {score}")
    print()


# Function to handle evaluation of a single model
def evaluate_model(model, X_test, y_test):
    accuracy = model.score(X_test, y_test)
    predictions = model.predict(X_test)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    
    return {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}

# Function to print test set results for volatility prediction models
def print_volatility_results(volatility_test_results):
    print("Volatility Prediction Test Set Results:")
    for model_name, results in volatility_test_results.items():
        print(model_name)
        for metric, score in results.items():
            print(f"{metric}: {score}")
        print()

# Function to print test set results for order execution models (if needed)
def print_order_execution_results(order_execution_test_results):
    if order_execution_test_results:
        print("Order Execution Test Set Results:")
        for model_name, results in order_execution_test_results.items():
            print(model_name)
            for metric, score in results.items():
                if metric == 'Confusion Matrix' or metric == 'Classification Report':
                    print(f"{metric}:\n{score}")
                else:
                    print(f"{metric}: {score}")
            print()

# Evaluate models on the test set
rf_results = evaluate_model(rf_model, X_test, y_test)
lr_results = evaluate_model(lr_model, X_test, y_test)
svm_results = evaluate_model(svm_model, X_test, y_test)

# Print test results for each model
volatility_test_results = {
    'Random Forest': rf_results,
    'Logistic Regression': lr_results,
    'SVM': svm_results
}

print_volatility_results(volatility_test_results)



# Extract configuration constants with default values
FOREX_SYMBOLS = config.get("FOREX_SYMBOLS", [])
START_DATE = config.get("START_DATE", "2010-01-01")
END_DATE = config.get("END_DATE", "2023-09-19")
SHORT_EMA_PERIOD = config.get("SHORT_EMA_PERIOD", 10)
LONG_EMA_PERIOD = config.get("LONG_EMA_PERIOD", 20)
RSI_OVERSOLD_THRESHOLD = config.get("RSI_OVERSOLD_THRESHOLD", 30)
RSI_OVERBOUGHT_THRESHOLD = config.get("RSI_OVERBOUGHT_THRESHOLD", 70)
MACD_SHORT_WINDOW = config.get("MACD_SHORT_WINDOW", 12)
MACD_LONG_WINDOW = config.get("MACD_LONG_WINDOW", 26)
EMAIL_NOTIFICATIONS = config.get("EMAIL_NOTIFICATIONS", True)
EMAIL_CONFIG = config.get("EMAIL_CONFIG", {})
ORDER_TYPE = config.get("ORDER_TYPE", "market")
MAX_RISK_PERCENT = config.get("MAX_RISK_PERCENT", 1)
KELLY_FRACTION = config.get("KELLY_FRACTION", 0.5)
TRAINING_PERIOD = config.get("TRAINING_PERIOD", 1000)
TEST_PERIOD = config.get("TEST_PERIOD", 100)
MIN_DATA_POINTS = config.get("MIN_DATA_POINTS", 1000)
REGIME_WINDOW = config.get("REGIME_WINDOW", 100)
CONFIDENCE_LEVEL = config.get("CONFIDENCE_LEVEL", 0.95)
LOG_FILE = config.get("LOG_FILE", "backtest.log")

# Maximum Risk Limit
MAX_RISK_PERCENT = 2 / 100  # 2 percent

# Kelly Criterion Constants
KELLY_FRACTION = 0.5

# Walk-Forward Validation Constants
TRAINING_PERIOD = 252
TEST_PERIOD = 63
MIN_DATA_POINTS = TRAINING_PERIOD + TEST_PERIOD

# Market Regime Detection Constants
REGIME_WINDOW = 50

# VaR Constants
CONFIDENCE_LEVEL = 0.95  # 95% confidence level

# Initialize slippage statistics dictionary

def configure_logging(log_file_path):
    try:
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        logging.config.dictConfig({
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'standard': {
                    'format': '%(asctime)s [%(levelname)s] - %(message)s',
                    'datefmt': '%Y-%m-%d %H:%M:%S'
                },
            },
            'handlers': {
                'file': {
                    'level': 'INFO',
                    'class': 'logging.handlers.RotatingFileHandler',
                    'filename': log_file_path,
                    'maxBytes': 1024,
                    'backupCount': 3,
                    'formatter': 'standard',
                },
            },
            'root': {
                'level': 'INFO',
                'handlers': ['file'],
            },
        })
    except Exception as e:
        logging.error(f"An error occurred while configuring logging: {str(e)}")

if __name__ == '__main__':
    log_file_path = '/path/to/your_log_file.log'
    configure_logging(log_file_path)

# Modularized Email Alert Function
def send_alert(subject: str, message: str):
    try:
        if EMAIL_NOTIFICATIONS:
            sender_email = EMAIL_CONFIG.get("sender_email")
            receiver_email = EMAIL_CONFIG.get("receiver_email")
            password = os.getenv("EMAIL_PASSWORD")  # Use environment variable for security

            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = receiver_email
            msg['Subject'] = subject

            msg.attach(MIMEText(message, 'plain'))

            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(sender_email, password)
            text = msg.as_string()
            server.sendmail(sender_email, receiver_email, text)
            server.quit()
            logging.info(f"Alert email sent: {subject}")
    except Exception as e:
        error_msg = f"Error sending alert email: {str(e)}"
        logging.error(error_msg)

def handle_error(msg):
    logging.error(f"Error: {msg}")
    raise ValueError(msg)

def calculate_ema(data: pd.DataFrame, period: int) -> pd.DataFrame:
    if not isinstance(period, int) or period <= 0:
        handle_error("Period must be a positive integer.")

    try:
        alpha = 2 / (period + 1)
        data['EMA'] = data['Close'].ewm(span=period, adjust=False).mean()
        return data
    except Exception as e:
        handle_error(f"EMA calculation: {str(e)}")

class SMACalculationError(Exception):
    pass

def calculate_sma(data, period):
    if not isinstance(data, pd.DataFrame):
        handle_error("Input 'data' should be a pandas DataFrame.")
    
    if not isinstance(period, int) or period <= 0:
        handle_error("Period should be a positive integer greater than zero.")
    
    if 'Close' not in data.columns:
        handle_error("Input data must contain a 'Close' column.")

    try:
        sma_values = data['Close'].rolling(window=period).mean()
        data = data.assign(SMA=sma_values)
        return data
    except Exception as e:
        handle_error(f"SMA calculation: {str(e)}")

def calculate_rsi(data, period=14):
    try:
        if not isinstance(data, pd.DataFrame):
            handle_error("Input 'data' should be a pandas DataFrame.")

        if not isinstance(period, int) or period <= 0:
            handle_error("Period should be a positive integer greater than zero.")

        if 'Close' not in data.columns:
            handle_error("Input data must contain a 'Close' column.")

        delta = data['Close'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        data['RSI'] = rsi
        return data.dropna()
    except (ValueError, KeyError) as e:
        handle_error(f"RSI calculation: {str(e)}")


class MACDCalculator:
    """
    Class for calculating Moving Average Convergence Divergence (MACD) and its signal line.
    """

    def __init__(self, data: pd.DataFrame, short_window: int, long_window: int, signal_line_period: int = 9):
        """
        Initialize MACDCalculator instance.

        Parameters:
        - data (pd.DataFrame): Input data with 'Close' column.
        - short_window (int): Short-term moving average window.
        - long_window (int): Long-term moving average window.
        - signal_line_period (int, optional): Period for the signal line (default is 9).
        """
        self.data = data
        self.short_window = short_window
        self.long_window = long_window
        self.signal_line_period = signal_line_period
        self.validate_input()

    def validate_input(self):
        """
        Validate input data and parameters.
        """
        if not isinstance(self.data, pd.DataFrame):
            raise ValueError("Input 'data' should be a pandas DataFrame.")

        if not all(isinstance(window, int) and window > 0 for window in [self.short_window, self.long_window, self.signal_line_period]):
            raise ValueError("Short and long windows, and signal line period should be positive integers greater than zero.")

        if 'Close' not in self.data.columns:
            raise ValueError("Input data must contain a 'Close' column.")

        if self.data['Close'].isnull().any():
            raise ValueError("Input data contains missing (NaN) values in the 'Close' column.")

    def calculate_ema(self, column_name, span):
        """
        Calculate Exponential Moving Average (EMA) for the specified column.

        Parameters:
        - column_name (str): Name of the column for EMA calculation.
        - span (int): Span parameter for EMA.

        Returns:
        - pd.Series: EMA values.
        """
        return self.data[column_name].ewm(span=span, adjust=False).mean()

    def calculate_macd(self):
        """
        Calculate MACD values and update the 'MACD' column in the data.
        """
        short_ema = self.calculate_ema('Close', self.short_window)
        long_ema = self.calculate_ema('Close', self.long_window)
        self.data['MACD'] = short_ema - long_ema
        return self

    def calculate_signal_line(self):
        """
        Calculate MACD Signal Line values and update the 'Signal_Line' column in the data.
        """
        self.data['Signal_Line'] = self.calculate_ema('MACD', self.signal_line_period)
        return self

    def calculate_macd_and_signal(self):
        """
        Calculate both MACD and Signal Line values.
        """
        self.calculate_macd()
        self.calculate_signal_line()
        return self.data

def configure_logging(log_filename, log_level=logging.INFO):
    """
    Configure logging settings.

    Parameters:
    - log_filename (str): Name of the log file.
    - log_level (int, optional): Logging level (default is INFO).
    """
    logging.basicConfig(filename=log_filename, level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def main():
    """
    Main function to demonstrate the usage of MACDCalculator.
    """
    data = pd.DataFrame({'Close': [100, 105, 110, 115, 120, 125, 130, 135, 140, 145]})
    short_window = 3
    long_window = 6
    signal_line_period = 9

    try:
        configure_logging('macd_log.txt', logging.INFO)
        macd_calculator = MACDCalculator(data, short_window, long_window, signal_line_period)
        result = macd_calculator.calculate_macd_and_signal()
        print(result)
    except ValueError as e:
        error_msg = f"Error calculating MACD: {str(e)}"
        logging.error(error_msg)
        raise ValueError(error_msg)

if __name__ == "__main__":
    main()


def calculate_candlestick_patterns(data):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input 'data' should be a pandas DataFrame.")

    if not all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
        raise ValueError("Input data must contain 'Open', 'High', 'Low', and 'Close' columns.")

    patterns = {
        'Bullish Engulfing': bullish_engulfing,
        'Bearish Engulfing': bearish_engulfing,
        'Doji': doji,
        'Morning Star': morning_star,
        'Evening Star': evening_star,
        'Shooting Star': shooting_star,
        'Dark Cloud Cover': dark_cloud_cover
        # Add more patterns and corresponding functions here
    }

    initialize_pattern_columns(data, patterns)

    calculate_and_update_patterns(data, patterns)

    return data

def initialize_pattern_columns(data, patterns):
    for pattern in patterns:
        data[pattern] = False
        data[f'{pattern} Confidence'] = 0.0

def calculate_and_update_patterns(data, patterns):
    for i in range(2, len(data)):
        for pattern, pattern_func in patterns.items():
            detected, confidence = pattern_func(data, i)
            data.at[i, pattern] = detected
            data.at[i, f'{pattern} Confidence'] = confidence

def bullish_engulfing(data, i):
    if (
        data['Close'].iloc[i] > data['Open'].iloc[i] and
        data['Open'].iloc[i - 1] > data['Close'].iloc[i - 1] and
        data['Close'].iloc[i - 1] > data['Open'].iloc[i] and
        data['Open'].iloc[i] < data['Close'].iloc[i - 1]
    ):
        confidence = 0.8
        return True, confidence
    else:
        return False, 0.0

def bearish_engulfing(data, i):
    if (
        data['Open'].iloc[i] > data['Close'].iloc[i] and
        data['Close'].iloc[i - 1] > data['Open'].iloc[i - 1] and
        data['Open'].iloc[i - 1] > data['Close'].iloc[i] and
        data['Close'].iloc[i] < data['Open'].iloc[i - 1]
    ):
        confidence = 0.7
        return True, confidence
    else:
        return False, 0.0


def doji(data, i):
    """
    Detects if a candle is a Doji pattern.

    Parameters:
    - data (pd.DataFrame): DataFrame containing financial data.
    - i (int): Index of the current candle.

    Returns:
    - tuple (bool, float): True if Doji pattern is detected, along with confidence score.
    """
    # Validate input parameters
    if not isinstance(data, pd.DataFrame) or not isinstance(i, int) or i < 2 or i >= len(data):
        raise ValueError("Invalid input parameters.")

    # Define a threshold for the body of the candle (a small value)
    body_threshold = 0.01 * (data['High'].iloc[i] - data['Low'].iloc[i])

    # Check if the current candle is a Doji
    if abs(data['Close'].iloc[i] - data['Open'].iloc[i]) < body_threshold:
        # Assign a confidence score for the detected pattern
        confidence = 0.6
        return True, confidence
    else:
        return False, 0.0


def morning_star(data, i):
    """
    Detects if a candle is a Morning Star pattern.

    Parameters:
    - data (pd.DataFrame): DataFrame containing financial data.
    - i (int): Index of the current candle.

    Returns:
    - tuple (bool, float): True if Morning Star pattern is detected, along with confidence score.
    """
    # Validate input parameters
    if not isinstance(data, pd.DataFrame) or not isinstance(i, int) or i < 3 or i >= len(data):
        raise ValueError("Invalid input parameters.")

    # Check if the current candle is a Morning Star pattern
    if (
        # Check the first candle
        data['Close'].iloc[i - 2] < data['Open'].iloc[i - 2] and
        data['Close'].iloc[i - 2] < data['Close'].iloc[i - 1] and

        # Check the second candle (the Doji or small body)
        data['Open'].iloc[i - 1] > data['Close'].iloc[i - 2] and
        data['Close'].iloc[i - 1] < data['Open'].iloc[i] and

        # Check the third candle
        data['Open'].iloc[i] < data['Close'].iloc[i - 2] and
        data['Close'].iloc[i] > data['Close'].iloc[i - 1]
    ):
        # Assign a confidence score for the detected pattern
        confidence = 0.9
        return True, confidence
    else:
        return False, 0.0


def evening_star(data, i):
    """
    Detects if a candle is an Evening Star pattern.

    Parameters:
    - data (pd.DataFrame): DataFrame containing financial data.
    - i (int): Index of the current candle.

    Returns:
    - tuple (bool, float): True if Evening Star pattern is detected, along with confidence score.
    """
    # Validate input parameters
    if not isinstance(data, pd.DataFrame) or not isinstance(i, int) or i < 3 or i >= len(data):
        raise ValueError("Invalid input parameters.")

    # Check if the current candle is an Evening Star pattern
    if (
        # Check the first candle
        data['Close'].iloc[i - 2] > data['Open'].iloc[i - 2] and
        data['Close'].iloc[i - 2] > data['Close'].iloc[i - 1] and

        # Check the second candle (the Doji or small body)
        data['Open'].iloc[i - 1] < data['Close'].iloc[i - 2] and
        data['Close'].iloc[i - 1] > data['Open'].iloc[i] and

        # Check the third candle
        data['Open'].iloc[i] > data['Close'].iloc[i - 2] and
        data['Close'].iloc[i] < data['Close'].iloc[i - 1]
    ):
        # Assign a confidence score for the detected pattern
        confidence = 0.7
        return True, confidence
    else:
        return False, 0.0


def shooting_star(data, i):
    """
    Detects if a candle is a Shooting Star pattern.

    Parameters:
    - data (pd.DataFrame): DataFrame containing financial data.
    - i (int): Index of the current candle.

    Returns:
    - tuple (bool, float): True if Shooting Star pattern is detected, along with confidence score.
    """
    # Validate input parameters
    if not isinstance(data, pd.DataFrame) or not isinstance(i, int) or i < 1 or i >= len(data):
        raise ValueError("Invalid input parameters.")

    # Check if the current candle is a Shooting Star pattern
    if (
        data['Open'].iloc[i] < data['Close'].iloc[i] and
        data['Close'].iloc[i] < data['Open'].iloc[i - 1] and
        (data['High'].iloc[i] - data['Close'].iloc[i]) > 2 * (data['Open'].iloc[i] - data['Low'].iloc[i])
    ):
        # Assign a confidence score for the detected pattern
        confidence = 0.6
        return True, confidence
    else:
        return False, 0.0

def dark_cloud_cover(data, i):
    if (
        data['Open'].iloc[i] < data['Close'].iloc[i - 1] and
        data['Close'].iloc[i] > (data['Close'].iloc[i - 1] + data['Open'].iloc[i - 1]) / 2 and
        data['Close'].iloc[i] < data['Open'].iloc[i - 1]
    ):
        confidence = 0.8
        return True, confidence
    else:
        return False, 0.0

def calculate_candlestick_patterns(data):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input 'data' should be a pandas DataFrame.")

    if not all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
        raise ValueError("Input data must contain 'Open', 'High', 'Low', and 'Close' columns.")

    patterns = {
        'Bullish Engulfing': bullish_engulfing,
        'Bearish Engulfing': bearish_engulfing,
        'Doji': doji,
        'Morning Star': morning_star,
        'Evening Star': evening_star,
        'Shooting Star': shooting_star,
        'Dark Cloud Cover': dark_cloud_cover
        # Add more patterns and corresponding functions here
    }

    initialize_pattern_columns(data, patterns)

    calculate_and_update_patterns(data, patterns)

    return data

def initialize_pattern_columns(data, patterns):
    for pattern in patterns:
        data[pattern] = False
        data[f'{pattern} Confidence'] = 0.0

def calculate_and_update_patterns(data, patterns):
    for i in range(2, len(data)):
        for pattern, pattern_func in patterns.items():
            detected, confidence = pattern_func(data, i)
            data.at[i, pattern] = detected
            data.at[i, f'{pattern} Confidence'] = confidence

def split_data(data, validation_size=0.2, random_state=42):
    features = ['Open', 'High', 'Low', 'Close', 'SMA', 'EMA', 'RSI', 'MACD']
    target = 'Trend_Direction'

    X = data[features]
    y = data[target]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_size, random_state=random_state)

    return X_train, X_val, y_train, y_val

# Example usage:
data = pd.DataFrame({
    'Open': [100, 110, 120, 130, 120, 110, 110],
    'High': [115, 125, 135, 140, 130, 125, 125],
    'Low': [95, 105, 115, 125, 115, 105, 105],
    'Close': [110, 120, 130, 125, 115, 115, 120]
})

data_with_patterns = calculate_candlestick_patterns(data)
print(data_with_patterns)

X_train, X_val, y_train, y_val = split_data(data)


# Define classification models
classification_models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'Support Vector Machine': SVC(random_state=42),
}

# Function to train and evaluate classification models
def train_and_evaluate_classification_models(X_train, y_train, X_val, y_val, models):
    """
    Train and evaluate classification models on the validation set.

    Parameters:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation target.
        models (dict): Dictionary of classification models to train and evaluate.

    Returns:
        dict: Dictionary of model evaluation metrics.
    """
    results = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        # Customize the evaluation metrics based on your needs
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='weighted')
        recall = recall_score(y_val, y_pred, average='weighted')
        f1 = f1_score(y_val, y_pred, average='weighted')

        results[model_name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
        }

    return results

# Train and evaluate classification models
classification_results = train_and_evaluate_classification_models(X_train, y_train, X_val, y_val, classification_models)

# Train the classifier
# Initialize a Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)

# Train the Random Forest Classifier on the entire dataset
rf_model.fit(X_train, y_train)

# Predict the target labels on the validation set
y_pred = rf_model.predict(X_val)


# Function to create and evaluate the model
def train_and_evaluate_model(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    return accuracy

# Define the objective function to minimize
def objective(hyperparameters):
    # Create a model using the hyperparameters
    model = create_model(model_name, hyperparameters)

    # Train and evaluate the model on the training and validation data
    score = train_and_evaluate_model(model, X_train, y_train, X_val, y_val)

    # Return the score to minimize (negative of accuracy, for example)
    return -score

# Specify the model name (e.g., 'Random Forest', 'Logistic Regression')
model_name = 'Random Forest'

# Define the hyperparameter search space for the chosen model
# (You may have different hyperparameters for each model)
hyperparameter_space = {
    'n_estimators': hp.quniform('n_estimators', 10, 100, 1),
    'max_depth': hp.quniform('max_depth', 1, 20, 1),
    'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 1),
    'max_features': hp.uniform('max_features', 0.1, 1.0),
}

# Use the Tree of Parzen Estimators (TPE) algorithm for optimization
print("Optimizing Hyperparameters:")
best_hyperparameters = fmin(fn=objective, space=hyperparameter_space, algo=tpe.suggest, max_evals=100)

# Print the best hyperparameters found by Bayesian optimization
print("\nBest hyperparameters:", best_hyperparameters)


# Function to create a machine learning model based on the model_name and hyperparameters
def create_model(model_name, hyperparameters):
    """
    Create a machine learning model.

    Parameters:
    - model_name (str): Name of the model ('Random Forest', 'Logistic Regression', 'Support Vector Machine').
    - hyperparameters (dict): Hyperparameters for the specified model.

    Returns:
    - model: Machine learning model instance.
    """
    try:
        if model_name == 'Random Forest':
            model = RandomForestClassifier(**hyperparameters)
        elif model_name == 'Logistic Regression':
            model = LogisticRegression(**hyperparameters)
        elif model_name == 'Support Vector Machine':
            model = SVC(**hyperparameters)
        # Add more model creation logic as needed
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        
        return model
    except Exception as e:
        print(f"Error creating model: {str(e)}")

# Rest of the code remains unchanged...

# Replace these with your actual data loading logic
X_train, y_train, X_val, y_val, X_test, y_test = load_your_data() 

# Example of an evaluation function for classification tasks (customize as needed)
def evaluate_classification_model(model, X_val, y_val):
    """
    Evaluate a classification model.

    Parameters:
    - model: Trained classification model.
    - X_val: Validation input data.
    - y_val: True labels for validation data.

    Returns:
    - dict: Dictionary containing evaluation metrics.
    """
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    return {
        'Accuracy': accuracy,
        'F1-Score': f1
    }

# Example of an evaluation function for regression tasks (customize as needed)
def evaluate_regression_model(model, X_val, y_val):
    """
    Evaluate a regression model.

    Parameters:
    - model: Trained regression model.
    - X_val: Validation input data.
    - y_val: True labels for validation data.

    Returns:
    - dict: Dictionary containing evaluation metrics.
    """
    y_pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    r2 = r2_score(y_val, y_pred)
    return {
        'RMSE': rmse,
        'R-squared': r2
    }


# Define the model name and hyperparameters
model_name = 'Random Forest'
hyperparameters = {
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 2
}

# Create the model
model = create_model(model_name, hyperparameters)

# Train and evaluate the model on the validation set
y_pred = train_and_evaluate_model(model, X_train, y_train, X_val, y_val)  # Ensure validation set is used during training

# Evaluate the model performance on the validation set
evaluation_metric = evaluate_model_performance(model, X_val, y_val, task_type='classification')

# Print the evaluation result
print(evaluation_metric)

# Initialize a Trials object to track the optimization process
trials = Trials()

# Perform Bayesian optimization to find the best hyperparameters
best = fmin(fn=objective,  # Objective function to minimize
            space=hyperparameter_space,  # Hyperparameter search space
            algo=tpe.suggest,  # Optimization algorithm (Tree-structured Parzen Estimator)
            max_evals=100,  # Maximum number of evaluations
            trials=trials,  # Optimization trials object
            verbose=1)  # Verbosity level (0: silent, 1: progress bar, 2: detailed)

# Retrieve the best hyperparameters and corresponding model name
best_hyperparameters = space_eval(hyperparameter_space, best)
best_model_name = best_hyperparameters['model']
best_hyperparameters = best_hyperparameters['hyperparameters']

# Create the best model with the optimal hyperparameters
best_model = create_model(best_model_name, best_hyperparameters)

# Train and evaluate the best model on the test set
y_pred = train_and_evaluate_model(best_model, X_train, y_train, X_test, y_test)

# Print the best hyperparameters and test accuracy
print("Best Model:", best_model_name)
print("Best Hyperparameters:", best_hyperparameters)
evaluation_metric = evaluate_model_performance(best_model, X_test, y_test, task_type='classification')
print("Test Accuracy:", evaluation_metric['Accuracy'])