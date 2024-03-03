import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import load_model
import plotly.graph_objs as go

# Function to calculate moving averages
def calculate_moving_average(data, window_size):
    return data.rolling(window=window_size).mean()

# Function to create dataset for prediction
def create_dataset(data, look_back=100):
    X = []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
    return np.array(X)

# Function to compare two stocks
def compare_stocks(stock_symbol1, stock_symbol2, start_date, end_date, selected_model):
    try:
        st.subheader(f"Comparison of {stock_symbol1} and {stock_symbol2}")

        # Load stock data for the first symbol
        stock_data1 = yf.download(stock_symbol1, start=start_date, end=end_date)

        # Load stock data for the second symbol
        stock_data2 = yf.download(stock_symbol2, start=start_date, end=end_date)

        # Calculate moving averages for the first symbol
        stock_data1['MA100'] = calculate_moving_average(stock_data1['Close'], 100)
        stock_data1['MA200'] = calculate_moving_average(stock_data1['Close'], 200)

        # Calculate moving averages for the second symbol
        stock_data2['MA100'] = calculate_moving_average(stock_data2['Close'], 100)
        stock_data2['MA200'] = calculate_moving_average(stock_data2['Close'], 200)

        # Display the data and charts side by side
        st.write('<style> .side-by-side { display: flex; }</style>', unsafe_allow_html=True)
        st.write('<div class="side-by-side">', unsafe_allow_html=True)

        # Display the data and charts for the first symbol
        st.write('<div style="width: 50%;">', unsafe_allow_html=True)
        st.subheader(f'Stock Data for {stock_symbol1}')
        st.write(stock_data1.head())
        st.subheader(f'Price vs MA100 for {stock_symbol1}')
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=stock_data1.index, y=stock_data1['Close'], mode='lines', name='Close Price'))
        fig1.add_trace(go.Scatter(x=stock_data1.index, y=stock_data1['MA100'], mode='lines', name='MA100'))
        st.plotly_chart(fig1)
        st.subheader(f'Price vs MA100 vs MA200 for {stock_symbol1}')
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=stock_data1.index, y=stock_data1['Close'], mode='lines', name='Close Price'))
        fig2.add_trace(go.Scatter(x=stock_data1.index, y=stock_data1['MA100'], mode='lines', name='MA100'))
        fig2.add_trace(go.Scatter(x=stock_data1.index, y=stock_data1['MA200'], mode='lines', name='MA200'))
        st.plotly_chart(fig2)
        st.write('</div>', unsafe_allow_html=True)

        # Display the data and charts for the second symbol
        st.write('<div style="width: 50%;">', unsafe_allow_html=True)
        st.subheader(f'Stock Data for {stock_symbol2}')
        st.write(stock_data2.head())
        st.subheader(f'Price vs MA100 for {stock_symbol2}')
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=stock_data2.index, y=stock_data2['Close'], mode='lines', name='Close Price'))
        fig3.add_trace(go.Scatter(x=stock_data2.index, y=stock_data2['MA100'], mode='lines', name='MA100'))
        st.plotly_chart(fig3)
        st.subheader(f'Price vs MA100 vs MA200 for {stock_symbol2}')
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=stock_data2.index, y=stock_data2['Close'], mode='lines', name='Close Price'))
        fig4.add_trace(go.Scatter(x=stock_data2.index, y=stock_data2['MA100'], mode='lines', name='MA100'))
        fig4.add_trace(go.Scatter(x=stock_data2.index, y=stock_data2['MA200'], mode='lines', name='MA200'))
        st.plotly_chart(fig4)
        st.write('</div>', unsafe_allow_html=True)

        st.write('</div>', unsafe_allow_html=True)

        # Forecasting
        st.subheader('30-Day Forecast')

        # Use the last 100 days of data for forecasting for the first symbol
        last_100_days1 = stock_data1['Close'].tail(100)
        last_100_days_scaled1 = MinMaxScaler().fit_transform(np.array(last_100_days1).reshape(-1, 1))

        # Use the last 100 days of data for forecasting for the second symbol
        last_100_days2 = stock_data2['Close'].tail(100)
        last_100_days_scaled2 = MinMaxScaler().fit_transform(np.array(last_100_days2).reshape(-1, 1))

        # Load models based on selection
        if selected_model == "Neural Network":
            model1 = load_model('KNN_model.keras')
            model2 = load_model('KNN_model.keras')
        elif selected_model == "Random Forest":
            model1 = load_model('random_forest_model.keras')
            model2 = load_model('random_forest_model.keras')
        elif selected_model == "Linear Regression":
            model1 = load_model('linear_regression_model.keras')
            model2 = load_model('linear_regression_model.keras')
        elif selected_model == "LSTM":
            lstm_model1 = load_model('lstm_model.keras')
            lstm_model2 = load_model('lstm_model.keras')

        # Forecast for the first symbol
        forecast_dates1 = [stock_data1.index[-1] + timedelta(days=i) for i in range(1, 31)]
        forecast1 = pd.DataFrame(index=forecast_dates1, columns=['Forecast'])
        for i in range(30):
            if selected_model != "LSTM":
                x_forecast1 = last_100_days_scaled1[-100:].reshape(1, -1)
                y_forecast1 = model1.predict(x_forecast1)
            else:
                x_forecast1 = last_100_days_scaled1.reshape(1, -1, 1)
                y_forecast1 = lstm_model1.predict(x_forecast1)
            forecast1.iloc[i] = MinMaxScaler().fit(stock_data1[['Close']]).inverse_transform(y_forecast1)[0][0]
            last_100_days_scaled1 = np.append(last_100_days_scaled1, y_forecast1)

        st.write(f"Forecast for {stock_symbol1}:")
        st.write(forecast1)

        # Forecast for the second symbol
        forecast_dates2 = [stock_data2.index[-1] + timedelta(days=i) for i in range(1, 31)]
        forecast2 = pd.DataFrame(index=forecast_dates2, columns=['Forecast'])
        for i in range(30):
            if selected_model != "LSTM":
                x_forecast2 = last_100_days_scaled2[-100:].reshape(1, -1)
                y_forecast2 = model2.predict(x_forecast2)
            else:
                x_forecast2 = last_100_days_scaled2.reshape(1, -1, 1)
                y_forecast2 = lstm_model2.predict(x_forecast2)
            forecast2.iloc[i] = MinMaxScaler().fit(stock_data2[['Close']]).inverse_transform(y_forecast2)[0][0]
            last_100_days_scaled2 = np.append(last_100_days_scaled2, y_forecast2)

        st.write(f"Forecast for {stock_symbol2}:")
        st.write(forecast2)

    except Exception as e:
        st.error(f"Error: {e}")

# Streamlit app
def main():
    st.sidebar.title('Stock Comparison App')

    # User input for first stock ticker symbol
    stock_symbol1 = st.sidebar.text_input('Enter First Stock Ticker Symbol (e.g., MSFT):')

    # User input for second stock ticker symbol
    stock_symbol2 = st.sidebar.text_input('Enter Second Stock Ticker Symbol (e.g., AAPL):')

    # Date range input
    start_date = st.sidebar.date_input('Select Start Date:', datetime.now() - timedelta(days=365))
    end_date = st.sidebar.date_input('Select End Date:', datetime.now())

    # Model selection
    selected_model = st.sidebar.radio("Select Model", ("Neural Network", "Random Forest", "Linear Regression", "LSTM"))

    # Compare stocks if both symbols are provided
    if stock_symbol1 and stock_symbol2:
        compare_stocks(stock_symbol1, stock_symbol2, start_date, end_date, selected_model)

if __name__ == '__main__':
    main()
