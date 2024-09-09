import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import yfinance as yf
from datetime import datetime, timedelta

# Define symbol for gold futures (e.g., June 2024 contract)
symbol = 'CL=F'

# Define the date range for the last month
end_date = datetime.now() + timedelta(1)
start_date = end_date - timedelta(days=30)

# Download daily data for the last month
data = yf.download(symbol, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval='1d')

# Drop rows with missing values
data = data.dropna()

# Ensure the index has a frequency of 'D' (daily)
data = data.asfreq('D')

# Define SARIMAX models with 7-day and 5-day seasonal periods
sarima_model_7 = SARIMAX(data['Close'], order=(1, 0, 1), seasonal_order=(1, 1, 1, 7))  # 7-day seasonal period
sarima_fit_7 = sarima_model_7.fit(disp=False)

sarima_model_5 = SARIMAX(data['Close'], order=(1, 0, 1), seasonal_order=(1, 1, 1, 5))  # 5-day seasonal period
sarima_fit_5 = sarima_model_5.fit(disp=False)

# Forecast steps for the next day
forecast_steps = 1  # Forecasting for 1 day ahead

# Predicted values for the next day
forecast_7 = sarima_fit_7.get_forecast(steps=forecast_steps)
predicted_price_7 = forecast_7.predicted_mean.iloc[0]

forecast_5 = sarima_fit_5.get_forecast(steps=forecast_steps)
predicted_price_5 = forecast_5.predicted_mean.iloc[0]

# Get the day of the week for the predicted day
predicted_date = data.index[-1] + timedelta(days=1)
predicted_day_name = predicted_date.strftime('%A')

# Print the predicted prices for the next day
print(f"Predicted Gold price for {predicted_day_name} (7-day seasonality): ${predicted_price_7:.2f} per ounce")
print(f"Predicted Gold price for {predicted_day_name} (5-day seasonality): ${predicted_price_5:.2f} per ounce")

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], marker='o', linestyle='-', color='b', label='Observed Prices (Last Month)')
plt.plot(predicted_date, predicted_price_7, 'ro', label=f'Predicted Price (7-day: {predicted_day_name})')
plt.plot(predicted_date, predicted_price_5, 'go', label=f'Predicted Price (5-day: {predicted_day_name})')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.title(f'**model7_1month**Observed and Predicted Prices for Gold\n(Predicted Day: {predicted_day_name})')
plt.legend()
plt.grid(True)

# Annotate the predicted prices with the day name
plt.annotate(f'{predicted_day_name}\n7-day: ${predicted_price_7:.2f}', 
             xy=(predicted_date, predicted_price_7), 
             xytext=(predicted_date + timedelta(days=2), predicted_price_7 + 10),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.annotate(f'{predicted_day_name}\n5-day: ${predicted_price_5:.2f}', 
             xy=(predicted_date, predicted_price_5), 
             xytext=(predicted_date + timedelta(days=2), predicted_price_5 - 10),
             arrowprops=dict(facecolor='green', shrink=0.05))

plt.tight_layout()
plt.show()
