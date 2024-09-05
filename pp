# Import necessary libraries and modules
import os
import logging
from logging.handlers import RotatingFileHandler
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed
import yfinance as yf
import backtrader as bt
from talib import RSI, MACD, CANDLE, EMA, SMA
from hyperopt import hp, tpe, fmin, space_eval
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import plot_model

# Import custom candlestick functions module
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
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        handler = RotatingFileHandler(log_file_path, maxBytes=1024, backupCount=3)
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    except FileNotFoundError as file_err:
        print(f"Error creating log directory: {file_err}")
    except Exception as e:
        print(f"An error occurred while configuring logging: {str(e)}")

# Example usage of the logging configuration
def example_usage_of_logging():
    """
    Example usage of logging with different log levels.

    Returns:
        None
    """
    try:
        logging.debug('This is a debug message')
        logging.info('This is an info message')
        logging.warning('This is a warning message')
        logging.error('This is an error message')
        logging.critical('This is a critical message')
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

# Function to download financial data using Yahoo Finance
def download_financial_data(symbols, start_date, end_date):
    """
    Download financial data using Yahoo Finance.

    Parameters:
        symbols (list): List of stock symbols.
        start_date (str): Start date for data download (YYYY-MM-DD).
        end_date (str): End date for data download (YYYY-MM-DD).

    Returns:
        pd.DataFrame or None: Downloaded financial data if successful, None otherwise.
    """
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
    
