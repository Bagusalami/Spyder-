#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 11:44:13 2024

@author: macbookpro
"""

import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import timedelta

class StockPredictor:
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
        """
        Mengunduh data harga saham dari Yahoo Finance
        """
        stock = yf.Ticker(self.ticker)
        df = stock.history(period=self.period)
        self.dates = df.index.to_pydatetime()  # Menyimpan tanggal
        self.prices = df['Close']  # Menyimpan harga penutupan

    def perform_linear_regression(self):
        """
        Melakukan regresi linear pada harga saham
        """
        # Mengonversi tanggal ke format ordinal (integer) untuk regresi
        X = np.array([date.toordinal() for date in self.dates]).reshape(-1, 1)
        y = np.array(self.prices)
        
        # Membuat dan melatih model regresi linear
        self.model = LinearRegression()
        self.model.fit(X, y)

    def predict_prices(self, years_forward=1):
        """
        Melakukan interpolasi dan prediksi harga saham untuk beberapa tahun ke depan
        """
        # Menggunakan model untuk prediksi pada data yang tersedia
        X = np.array([date.toordinal() for date in self.dates]).reshape(-1, 1)
        self.interpolated_prices = self.model.predict(X)
        
        # Prediksi untuk beberapa tahun ke depan
        self.future_dates = [self.dates[-1] + timedelta(days=i) for i in range(1, 365 * years_forward + 1)]
        X_future = np.array([date.toordinal() for date in self.future_dates]).reshape(-1, 1)
        self.predicted_prices = self.model.predict(X_future)

    def plot_results(self):
        """
        Menampilkan hasil visualisasi harga saham
        """
        plt.figure(figsize=(10,6))

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
        """
        Menggabungkan seluruh proses mulai dari unduh data, regresi, prediksi, hingga plotting
        """
        self.download_stock_data()
        self.perform_linear_regression()
        self.predict_prices()
        self.plot_results()

# Main function
if __name__ == "__main__":
    # Menginstansiasi objek StockPredictor dengan ticker AAPL
    stock_predictor = StockPredictor("AAPL")
    stock_predictor.run()
