import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import joblib

# Function to calculate moving average
def calculate_moving_average(data, window_size):
    return data.rolling(window=window_size).mean()

# Function to forecast next 7 days' stock prices using Keras model
def forecast_next_7_days_keras(data):
    try:
        keras_model = load_model('lstm_model.keras')  # Load LSTM model for forecasting
        last_100_days = data[-100:].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(last_100_days)
        scaled_last_100_days = scaler.transform(last_100_days)
        x_pred = scaled_last_100_days.reshape(1, 100, 1)
        forecasts = []
        for _ in range(7):
            next_day_pred = keras_model.predict(x_pred)[0, 0]
            forecasts.append(next_day_pred)
            x_pred = np.roll(x_pred, -1)
            x_pred[0, -1, 0] = next_day_pred
        forecasts = np.array(forecasts).reshape(-1, 1)
        return scaler.inverse_transform(forecasts).flatten()
    except Exception as e:
        st.error(f"Error in forecasting using Keras model: {str(e)}")
        return []

# Function to forecast next 7 days' stock prices using Linear Regression model
def forecast_next_7_days_linear_regression(data):
    try:
        linear_regression_model = joblib.load('linear_regression_model.h5')  # Load Linear Regression model for forecasting
        return linear_regression_model.predict(data[-7:].values.reshape(1, -1))
    except Exception as e:
        st.error(f"Error in forecasting using Linear Regression model: {str(e)}")
        return []

# Function to forecast next 7 days' stock prices using Random Forest model
def forecast_next_7_days_random_forest(data):
    try:
        random_forest_model = joblib.load('random_forest_model.h5')  # Load Random Forest model for forecasting
        return random_forest_model.predict(data[-7:].values.reshape(1, -1))
    except Exception as e:
        st.error(f"Error in forecasting using Random Forest model: {str(e)}")
        return []

# Function to forecast next 7 days' stock prices using ARIMA model
def forecast_next_7_days_arima(data):
    try:
        arima_model = joblib.load('arima_model.pkl')  # Load ARIMA model for forecasting
        forecast = arima_model.predict(n_periods=7)
        return forecast
    except Exception as e:
        st.error(f"Error in forecasting using ARIMA model: {str(e)}")
        return []

# Streamlit UI
st.title('Stock Market Predictor')

# Sidebar: Input parameters
st.sidebar.subheader('Input Parameters')
stock = st.sidebar.text_input('Enter Stock Symbol', 'MSFT')
start_date = st.sidebar.date_input('Select Start Date', pd.to_datetime('2000-01-01'))
end_date = st.sidebar.date_input('Select End Date', pd.to_datetime('today'))

# Fetch stock data
data = yf.download(stock, start=start_date, end=end_date)
data.reset_index(inplace=True)

# Display stock data
st.subheader('Stock Data')
st.write(data)

# Calculate moving averages
ma_100_days = calculate_moving_average(data['Close'], 100)
ma_200_days = calculate_moving_average(data['Close'], 200)

# Plot moving averages
st.subheader('Moving Average Plots')
fig_ma100 = go.Figure()
fig_ma100.add_trace(go.Scatter(x=data['Date'], y=ma_100_days, mode='lines', name='MA100'))
fig_ma100.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close Price'))
fig_ma100.update_layout(title='Price vs MA100', xaxis_title='Date', yaxis_title='Price')
st.plotly_chart(fig_ma100)

fig_ma200 = go.Figure()
fig_ma200.add_trace(go.Scatter(x=data['Date'], y=ma_100_days, mode='lines', name='MA100', line=dict(color='red')))
fig_ma200.add_trace(go.Scatter(x=data['Date'], y=ma_200_days, mode='lines', name='MA200', line=dict(color='blue')))
fig_ma200.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close Price', line=dict(color='green')))
fig_ma200.update_layout(title='Price vs MA100 vs MA200', xaxis_title='Date', yaxis_title='Price')
st.plotly_chart(fig_ma200)

# Machine Learning Model Selection
ml_models = {
    'Keras LSTM Model': forecast_next_7_days_keras,
    'Linear Regression Model': forecast_next_7_days_linear_regression,
    'Random Forest Model': forecast_next_7_days_random_forest,
    'ARIMA Model': forecast_next_7_days_arima
}

selected_model = st.selectbox('Select Model', list(ml_models.keys()))

# Forecast next 7 days' stock prices
forecasted_prices = ml_models[selected_model](data)

# Display forecasted prices
st.subheader('Next 7 Days Forecasted Close Prices')
forecast_dates = pd.date_range(start=data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=7)
forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecasted Close Price': forecasted_prices})
st.write(forecast_df)
