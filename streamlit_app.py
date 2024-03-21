import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import plotly.graph_objs as go
import requests

# Function to calculate moving averages
def calculate_moving_average(data, window_size):
    return data.rolling(window=window_size).mean()

# Function to create dataset for prediction
def create_dataset(data, look_back=100):
    X = []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
    return np.array(X)

# Function to download model file
def download_model(model_url, model_filename):
    response = requests.get(model_url)
    with open(model_filename, 'wb') as f:
        f.write(response.content)

# Streamlit app
def main():
    st.sidebar.title('Stock Price Forecasting App')
    st.sidebar.markdown('Copyright by Rajdeep Sarkar')

    # User input for stock ticker symbol
    stock_symbol = st.sidebar.text_input('Enter Stock Ticker Symbol (e.g., MSFT):')

    # Date range input
    start_date = st.sidebar.date_input('Select Start Date:', datetime.now() - timedelta(days=365))
    end_date = st.sidebar.date_input('Select End Date:', datetime.now())

    # Model selection
    selected_model = st.sidebar.radio("Select Model", ("Neural Network", "Random Forest"))

    # Load stock data
    if stock_symbol:
        try:
            stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
            st.subheader('Stock Data')
            st.write(stock_data.head(50))  # Display first 50 rows
            st.write("...")  # Inserting an ellipsis for large datasets

            # Calculate moving averages
            stock_data['MA100'] = calculate_moving_average(stock_data['Close'], 100)
            stock_data['MA200'] = calculate_moving_average(stock_data['Close'], 200)

            # Plot stock data with moving average
            st.subheader('Price vs MA100')
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close Price'))
            fig1.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MA100'], mode='lines', name='MA100'))
            st.plotly_chart(fig1)

            # Plot stock data with moving averages
            st.subheader('Price vs MA100 vs MA200')
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close Price'))
            fig2.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MA100'], mode='lines', name='MA100'))
            fig2.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MA200'], mode='lines', name='MA200'))
            st.plotly_chart(fig2)

            # Additional plots for the selected stock
            st.subheader('Additional Plots')
            # Candlestick chart
            candlestick = go.Candlestick(x=stock_data.index,
                                         open=stock_data['Open'],
                                         high=stock_data['High'],
                                         low=stock_data['Low'],
                                         close=stock_data['Close'],
                                         name='Candlestick')
            candlestick_layout = go.Layout(title='Candlestick Chart')
            candlestick_fig = go.Figure(data=candlestick, layout=candlestick_layout)
            st.plotly_chart(candlestick_fig)

            # Volume plot
            volume_fig = go.Figure()
            volume_fig.add_trace(go.Bar(x=stock_data.index, y=stock_data['Volume'], name='Volume'))
            volume_fig.update_layout(title='Volume Plot')
            st.plotly_chart(volume_fig)

            # Load trained model based on selection
            if selected_model == "Neural Network":
                model_url = "https://github.com/rajdeepUWE/stock_market_forecast/raw/master/KNN_model.h5"
                model_filename = "KNN_model.h5"
            elif selected_model == "Random Forest":
                model_url = "https://github.com/rajdeepUWE/stock_market_forecast/raw/master/random_forest_model.h5"
                model_filename = "random_forest_model.h5"

            # Download model file
            download_model(model_url, model_filename)

            # Load model
            model = load_model(model_filename)

            # Compile the model with appropriate optimizer and loss function
            model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

            # Scale data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(np.array(stock_data['Close']).reshape(-1, 1))

            # Prepare data for prediction
            x_pred = create_dataset(scaled_data)

            # Predict stock prices
            y_pred = model.predict(x_pred)
            y_pred = scaler.inverse_transform(y_pred)

            # Plot original vs predicted prices
            st.subheader('Original vs Predicted Prices')
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Original Price'))
            fig3.add_trace(go.Scatter(x=stock_data.index[100:], y=y_pred.flatten(), mode='lines', name='Predicted Price'))
            st.plotly_chart(fig3)

            # Forecasting
            forecast_dates = [stock_data.index[-1] + timedelta(days=i) for i in range(1, 31)]
            forecast = pd.DataFrame(index=forecast_dates, columns=['Forecast'])

            # Use the last 100 days of data for forecasting
            last_100_days = stock_data['Close'].tail(100)
            last_100_days_scaled = scaler.transform(np.array(last_100_days).reshape(-1, 1))

            for i in range(30):
                x_forecast = last_100_days_scaled[-100:].reshape(1, -1)
                y_forecast = model.predict(x_forecast)
                forecast.iloc[i] = scaler.inverse_transform(y_forecast)[0][0]
                last_100_days_scaled = np.append(last_100_days_scaled, y_forecast)

            st.subheader('30-Day Forecast')
            st.write(forecast)

        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == '__main__':
    main()
