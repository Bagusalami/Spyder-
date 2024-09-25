#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 13:52:39 2024

@author: macbookpro
"""

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# 1. Mengunduh data harga saham dari Yahoo Finance
def download_stock_data(ticker, period='5y'):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    return df['Close']

# 2. Melakukan regresi linear
def perform_linear_regression(dates, prices):
    # Mengonversi tanggal ke format ordinal (integer)
    X = np.array([date.toordinal() for date in dates]).reshape(-1, 1)
    y = np.array(prices)
    
    # Model regresi linear
    model = LinearRegression()
    model.fit(X, y)
    
    return model

# 3. Interpolasi dan Prediksi
def predict_prices(model, dates, years_forward=1):
    # Menggunakan model untuk prediksi pada data yang tersedia
    X = np.array([date.toordinal() for date in dates]).reshape(-1, 1)
    interpolated_prices = model.predict(X)
    
    # Prediksi 1 tahun ke depan
    future_dates = [dates[-1] + timedelta(days=i) for i in range(1, 365 * years_forward + 1)]
    X_future = np.array([date.toordinal() for date in future_dates]).reshape(-1, 1)
    predicted_prices = model.predict(X_future)
    
    return interpolated_prices, future_dates, predicted_prices

# 4. Plotting hasil
def plot_results(dates, actual_prices, interpolated_prices, future_dates, predicted_prices):
    plt.figure(figsize=(10,6))
    
    # Plot harga aktual
    plt.plot(dates, actual_prices, label="Actual Prices", color='blue')
    
    # Plot interpolasi (fit regresi)
    plt.plot(dates, interpolated_prices, label="Interpolated Trend", color='green', linestyle='--')
    
    # Plot prediksi
    plt.plot(future_dates, predicted_prices, label="Predicted Prices (1 year ahead)", color='red', linestyle='--')
    
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.title("Stock Price Trend and Prediction")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Main function
def main():
    # 1. Download data saham
    ticker = "AAPL"  # Ganti dengan ticker saham yang diinginkan
    stock_prices = download_stock_data(ticker)
    dates = stock_prices.index.to_pydatetime()

    # 2. Melakukan regresi linear
    model = perform_linear_regression(dates, stock_prices)
    
    # 3. Prediksi harga 1 tahun ke depan
    interpolated_prices, future_dates, predicted_prices = predict_prices(model, dates)
    
    # 4. Plot hasil
    plot_results(dates, stock_prices, interpolated_prices, future_dates, predicted_prices)

# Memanggil main function
main()
