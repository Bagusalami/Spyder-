#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 11:42:54 2024

@author: macbookpro
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import timedelta

class StockPricePredictor:
    def __init__(self, ticker, period='5y'):
        self.ticker = ticker
        self.period = period
        self.dates = None
        self.prices = None
        self.model = None
        self.interpolated_prices = None
        self.future_dates = None
        self.predicted_prices = None

    def download_stock_data(self):
        stock = yf.Ticker(self.ticker)
        df = stock.history(period=self.period)
        self.dates = df.index.to_pydatetime()  # Menyimpan tanggal
        self.prices = df['Close']  # Menyimpan harga penutupan

    def perform_linear_regression(self):
        X = np.array([date.toordinal() for date in self.dates]).reshape(-1, 1)
        y = np.array(self.prices)

        self.model = LinearRegression()
        self.model.fit(X, y)

    def predict_prices(self, years_forward=1):
        # Interpolasi data harga saham dengan model regresi
        X = np.array([date.toordinal() for date in self.dates]).reshape(-1, 1)
        self.interpolated_prices = self.model.predict(X)

        # Prediksi harga saham untuk 1 tahun ke depan
        self.future_dates = [self.dates[-1] + timedelta(days=i) for i in range(1, 365 * years_forward + 1)]
        X_future = np.array([date.toordinal() for date in self.future_dates]).reshape(-1, 1)
        self.predicted_prices = self.model.predict(X_future)

    def plot_results(self):
        plt.figure(figsize=(10, 6))

        # Plot harga aktual
        plt.plot(self.dates, self.prices, label="Actual Prices", color='blue')

        # Plot interpolasi (fit regresi)
        plt.plot(self.dates, self.interpolated_prices, label="Interpolated Trend", color='green', linestyle='--')

        # Plot prediksi
        plt.plot(self.future_dates, self.predicted_prices, label="Predicted Prices (1 year ahead)", color='red', linestyle='--')

        plt.xlabel("Date")
        plt.ylabel("Stock Price")
        plt.title("Stock Price Trend and Prediction")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def run(self):
        self.download_stock_data()
        self.perform_linear_regression()
        self.predict_prices()
        self.plot_results()

# Memanggil class dan menjalankan prediksi
if __name__ == "__main__":
    predictor = StockPricePredictor("AAPL")  # Ganti dengan ticker saham yang diinginkan
    predictor.run()
