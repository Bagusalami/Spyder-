#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 13:55:30 2024

@author: macbookpro
"""

import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

start_date = "2023-01-01"
end_date = "2024-09-01"

data = yf.download("AAPL", start=start_date, end=end_date)

date_array = data.index.to_numpy()
data_array = data.to_numpy()

date = data.index

plt.figure(figsize=(15,6))

plt.plot(data.index, data["Adj Close"], label="Adj Close")
# plt.plot(data.index, data["High"], label="High")
# plt.plot(data.index, data["Low"], label="Low")
plt.grid(linestyle=":")
plt.ylabel("Price ($)")

plt.title(f"Apple stock price from {start_date} to {end_date} and prediction based on its trend")

"""
linear regression with sklearn

conda install scikit-learn
"""

from sklearn.linear_model import LinearRegression

data['dates_numeric'] = data.index.map(pd.Timestamp.timestamp)


# Features (X) and target (y)
x = data['dates_numeric'].values.reshape(-1, 1)  # Reshape for sklearn
y = data['Adj Close']

# Fit the linear regression model
model = LinearRegression()
model.fit(x, y)

# Predict values
y_pred = model.predict(x)

# plot interpolation
plt.plot(data.index, y_pred, label="Interpolation", c="orange", linestyle="-")
plt.legend()

# save plot
plt.savefig("Apple_Stock_Price.png", dpi=300)