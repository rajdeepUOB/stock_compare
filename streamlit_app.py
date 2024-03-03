import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import load_model
import plotly.graph_objs as go
import requests
import talib

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

            # Volume Plot
            st.subheader('Volume')
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(x=stock_data.index, y=stock_data['Volume'], name='Volume'))
            st.plotly_chart(fig3)

            # Volatility Plot
            st.subheader('Volatility')
            daily_returns = stock_data['Close'].pct_change()
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=stock_data.index, y=daily_returns, mode='lines', name='Daily Returns'))
            st.plotly_chart(fig4)

            # Candlestick Chart
            st.subheader('Candlestick Chart')
            fig5 = go.Figure(data=[go.Candlestick(x=stock_data.index,
                                                  open=stock_data['Open'],
                                                  high=stock_data['High'],
                                                  low=stock_data['Low'],
                                                  close=stock_data['Close'])])
            st.plotly_chart(fig5)

            # Technical Indicators
            st.subheader('Technical Indicators')
            rsi = talib.RSI(stock_data['Close'])
            macd, macdsignal, _ = talib.MACD(stock_data['Close'])
            upper, middle, lower = talib.BBANDS(stock_data['Close'])
            fig6 = go.Figure()
            fig6.add_trace(go.Scatter(x=stock_data.index, y=rsi, mode='lines', name='RSI'))
            fig6.add_trace(go.Scatter(x=stock_data.index, y=macd, mode='lines', name='MACD'))
            fig6.add_trace(go.Scatter(x=stock_data.index, y=upper, mode='lines', name='Upper BBand'))
            fig6.add_trace(go.Scatter(x=stock_data.index, y=middle, mode='lines', name='Middle BBand'))
            fig6.add_trace(go.Scatter(x=stock_data.index, y=lower, mode='lines', name='Lower BBand'))
            st.plotly_chart(fig6)

            # Correlation Heatmap
            st.subheader('Correlation Heatmap')
            correlation_data = yf.download(stock_symbol, start=start_date - timedelta(days=365), end=end_date)
            correlation_heatmap = correlation_data['Close'].pct_change().corr()
            fig7 = go.Figure(data=go.Heatmap(z=correlation_heatmap.values,
                                              x=correlation_heatmap.index,
                                              y=correlation_heatmap.columns))
            st.plotly_chart(fig7)

            # Histogram of Returns
            st.subheader('Histogram of Returns')
            fig8 = go.Figure()
            fig8.add_trace(go.Histogram(x=daily_returns, nbinsx=50, name='Returns'))
            st.plotly_chart(fig8)

            # Price Distribution Plot
            st.subheader('Price Distribution Plot')
            fig9 = go.Figure()
            fig9.add_trace(go.Histogram(x=stock_data['Close'], nbinsx=50, name='Price'))
            st.plotly_chart(fig9)

            # Seasonal Decomposition
            st.subheader('Seasonal Decomposition')
            decomposition = seasonal_decompose(stock_data['Close'], model='additive', period=30)
            fig10 = go.Figure()
            fig10.add_trace(go.Scatter(x=stock_data.index, y=decomposition.trend, mode='lines', name='Trend'))
            fig10.add_trace(go.Scatter(x=stock_data.index, y=decomposition.seasonal, mode='lines', name='Seasonal'))
            fig10.add_trace(go.Scatter(x=stock_data.index, y=decomposition.resid, mode='lines', name='Residual'))
            st.plotly_chart(fig10)

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
            fig11 = go.Figure()
            fig11.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Original Price'))
            fig11.add_trace(go.Scatter(x=stock_data.index[100:], y=y_pred.flatten(), mode='lines', name='Predicted Price'))
            st.plotly_chart(fig11)

            # Evaluation metrics
            y_true = stock_data['Close'].values[100:]
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)

            st.subheader('Model Evaluation')
            st.write(f'Mean Absolute Error (MAE): {mae:.2f}')
            st.write(f'Mean Squared Error (MSE): {mse:.2f}')

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
