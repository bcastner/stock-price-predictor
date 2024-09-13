# use the yfinance library to collect stock market data for a company
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import keras as ks
from keras import layers


# Download historical stock data for Apple (AAPL)
stock_data = yf.download('AAPL', start='2015-01-01', end='2023-01-01')

# View the first few rows of the data
print(stock_data.head())

# Plot the closing price
plt.figure(figsize=(12, 6))
plt.plot(stock_data['Close'], label='Apple Close Price')
plt.title('Apple Stock Closing Price Over Time')
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')
plt.legend()
plt.show()

# Preprocess the data to make it suitable for time-series forecasting
# Since LSTMs perform better with scaled data, we use MinMaxScaler
# LSTMs require sequences of data to predict the next value, so we will split the data
# into sequences of 'n' previous days' to predict the next day's price
# Use only the 'Close' column for prediction
close_prices = stock_data['Close'].values

# Scale the data between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices.reshape(-1, 1))


# Create sequences of 60 dats of data to predict the 61st day's closing price
def create_sequences(data, sequence_length):
    sequences = []
    labels = []
    for i in range(sequence_length, len(data)):
        sequences.append(data[i-sequence_length:i, 0])
        labels.append(data[i, 0])
    return np.array(sequences), np.array(labels)


sequence_length = 60
X, y = create_sequences(scaled_data, sequence_length)

# Split the data into training and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape X for LSTM (samples, time steps, features)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))

# Build and train the LSTM model using Keras
model = ks.Sequential()

# LSTM layer
model.add(ks.layers.LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(ks.layers.Dropout(0.2))    # Add dropout for regularization

# Add another LSTM layer
model.add(ks.layers.LSTM(units=50, return_sequences=False))
model.add(ks.layers.Dropout(0.2))

# Dense layer
model.add(ks.layers.Dense(units=25))

# Output layer (predicted next closing price)
model.add(ks.layers.Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, batch_size=64, epochs=10)

# Once the model is trained, we can make predictions to test data
predictions = model.predict(X_test)

# Inverse scale the predictions to get them back to the original range
predictions = scaler.inverse_transform(predictions.reshape(-1, 1))

# Inverse scale the test data to compare with actual values
y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Evaluate and visualize the results
# Plot the actual vs predicted stock prices
plt.figure(figsize=(12, 6))
plt.plot(y_test_unscaled, label='Actual Price')
plt.plot(predictions, label='Predicted Price')
plt.title('Apple Stock Price Prediction')
plt.xlabel('Days')
plt.ylabel('Price USD ($)')
plt.legend()
plt.show()

# We can also calculate the Mean Squared Error (MSE)
mse = mean_squared_error(y_test_unscaled, predictions)
print("*" * 50)
print(f"Mean Squared Error: {mse}")
