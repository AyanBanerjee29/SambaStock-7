# SAMBA Stock Price Prediction System

This project implements the **SAMBA (State-space Mamba with Graph Neural Networks)** architecture for stock price forecasting. It includes a complete pipeline for downloading financial data, training a model with hyperparameter tuning, evaluating performance, and generating future stock price predictions.

## 📋 Table of Contents
1. [Installation](#1-installation)
2. [Project Structure](#2-project-structure)
3. [Step 1: Create Dataset](#3-step-1-create-dataset)
4. [Step 2: Training & Development](#4-step-2-training--development)
5. [Step 3: Testing & Evaluation](#5-step-3-testing--evaluation)
6. [Step 4: Production Training](#6-step-4-production-training)
7. [Step 5: Future Prediction](#7-step-5-future-prediction)

---

## 1. Installation

Ensure you have Python installed (3.8+ recommended).

1. **Clone the repository** (if applicable) or navigate to your project folder.
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
2. Project Structure
Plaintext

.
├── create_feature_dataset.py   # Script to download and process raw data
├── main.py                     # Main entry point for Train/Test/Production
├── predict_future.py           # Script to forecast next 7 days
├── paper_config.py             # Configuration matching the SAMBA paper
├── requirements.txt            # Python dependencies
├── Dataset/                    # Folder where processed CSVs are stored
├── final_model_outputs/        # Folder where models and params are saved
├── models/                     # SAMBA and Mamba model definitions
├── trainer/                    # Training loop implementation
└── utils/                      # Utility functions (metrics, logging, etc.)
3. Step 1: Create Dataset
Before training, you must download the raw financial data and generate the feature dataset. This script downloads data from Yahoo Finance (tickers like ^NSEI, ^GSPC, INR=X, etc.) and calculates technical indicators (RSI, MACD, Bollinger Bands).

Run the command:

Bash

python create_feature_dataset.py
Output: Creates a file at Dataset/NIFTY50_features_wide.csv.

Note: This dataset contains NIFTY 50 data combined with global indices and technical indicators.

4. Step 2: Training & Development
This mode splits the data (80% Dev, 20% Test). It performs hyperparameter tuning on the Dev set, finds the best parameters, and trains a validation model.

Run the command:

Bash

python main.py --mode train
What this does:

Loads data from Dataset/NIFTY50_features_wide.csv.

Splits data into Dev (80%) and Test (20%).

Runs a grid search for hyperparameters (Learning Rate, Hidden Dim, etc.).

Saves the best parameters to final_model_outputs/best_params.json.

Trains a model on the 80% Dev set and saves it to final_model_outputs/dev_samba_model.pth.

5. Step 3: Testing & Evaluation
Once you have a trained "Dev" model, you can evaluate its performance on the unseen 20% test set.

Run the command:

Bash

python main.py --mode test
What this does:

Loads the model from final_model_outputs/dev_samba_model.pth.

Evaluates it on the held-out 20% Test data.

Prints metrics: MAE, RMSE, and IC (Information Coefficient).

Generates and saves prediction plots (e.g., test_plot_day_1.png) to final_model_outputs/.

6. Step 4: Production Training
When you are satisfied with the model's performance, train a "Production" model using 100% of the available data. This model is intended for making real future predictions.

Run the command:

Bash

python main.py --mode production
What this does:

Loads the best hyperparameters found in Step 2.

Trains the SAMBA model on the entire dataset.

Saves the final model to final_model_outputs/production_samba_model.pth.

7. Step 5: Future Prediction
Use the production model to forecast stock prices for the next 7 business days (or configured horizon).

Run the command:

Bash

python predict_future.py
What this does:

Loads final_model_outputs/production_samba_model.pth.

Loads the latest data from Dataset/NIFTY50_features_wide.csv.

Takes the most recent 60-day window (lookback) from the dataset.

Predicts the closing price for the next 7 days.

Prints the forecast dates and prices to the console.

Saves the results to final_model_outputs/next_week_forecast.csv.
