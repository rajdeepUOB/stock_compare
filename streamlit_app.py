import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, LSTM
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

# Function to compare stocks
def compare_stocks(stock_symbol1, stock_symbol2, start_date, end_date, selected_model):
    try:
        # Download stock data for first symbol
        stock_data1 = yf.download(stock_symbol1, start=start_date, end=end_date)
        st.subheader(f'Stock Data for {stock_symbol1}')
        st.write(stock_data1.head(50))  # Display first 50 rows

        # Calculate moving averages for first symbol
        stock_data1['MA100'] = calculate_moving_average(stock_data1['Close'], 100)
        stock_data1['MA200'] = calculate_moving_average(stock_data1['Close'], 200)

        # Plot stock data with moving averages for first symbol
        st.subheader(f'Price vs MA100 for {stock_symbol1}')
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=stock_data1.index, y=stock_data1['Close'], mode='lines', name='Close Price'))
        fig1.add_trace(go.Scatter(x=stock_data1.index, y=stock_data1['MA100'], mode='lines', name='MA100'))
        st.plotly_chart(fig1)

        # Plot stock data with moving averages for first symbol
        st.subheader(f'Price vs MA100 vs MA200 for {stock_symbol1}')
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=stock_data1.index, y=stock_data1['Close'], mode='lines', name='Close Price'))
        fig2.add_trace(go.Scatter(x=stock_data1.index, y=stock_data1['MA100'], mode='lines', name='MA100'))
        fig2.add_trace(go.Scatter(x=stock_data1.index, y=stock_data1['MA200'], mode='lines', name='MA200'))
        st.plotly_chart(fig2)

        # Load trained model based on selection
        if selected_model == "Neural Network":
            model1 = load_model('KNN_model.keras')
        elif selected_model == "Random Forest":
            model1 = load_model('random_forest_model.keras')
        elif selected_model == "Linear Regression":
            model1 = load_model('linear_regression_model.keras')
        elif selected_model == "LSTM":
            lstm_model1 = load_model('lstm_model.keras')

        # Scale data for first symbol
        scaler1 = MinMaxScaler(feature_range=(0, 1))
        scaled_data1 = scaler1.fit_transform(np.array(stock_data1['Close']).reshape(-1, 1))

        # Prepare data for prediction for first symbol
        x_pred1 = create_dataset(scaled_data1)

        # Predict stock prices for first symbol
        if selected_model != "LSTM":
            y_pred1 = model1.predict(x_pred1)
            y_pred1 = scaler1.inverse_transform(y_pred1)
        else:
            lstm_x_pred1 = create_dataset(scaled_data1)
            lstm_y_pred1 = lstm_model1.predict(lstm_x_pred1)
            lstm_y_pred1 = scaler1.inverse_transform(lstm_y_pred1)

        # Plot original vs predicted prices for first symbol
        st.subheader(f'Original vs Predicted Prices for {stock_symbol1}')
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=stock_data1.index, y=stock_data1['Close'], mode='lines', name='Original Price'))
        if selected_model != "LSTM":
            fig3.add_trace(go.Scatter(x=stock_data1.index[100:], y=y_pred1.flatten(), mode='lines', name='Predicted Price'))
        else:
            fig3.add_trace(go.Scatter(x=stock_data1.index[100:], y=lstm_y_pred1.flatten(), mode='lines', name='LSTM Predicted Price'))
        st.plotly_chart(fig3)

        # Evaluation metrics for first symbol
        y_true1 = stock_data1['Close'].values[100:]
        if selected_model != "LSTM":
            mae1 = mean_absolute_error(y_true1, y_pred1)
            mse1 = mean_squared_error(y_true1, y_pred1)
        else:
            mae1 = mean_absolute_error(y_true1, lstm_y_pred1)
            mse1 = mean_squared_error(y_true1, lstm_y_pred1)

        st.subheader(f'Model Evaluation for {stock_symbol1}')
        st.write(f'Mean Absolute Error (MAE): {mae1:.2f}')
        st.write(f'Mean Squared Error (MSE): {mse1:.2f}')

        # Forecasting for first symbol
        forecast_dates1 = [stock_data1.index[-1] + timedelta(days=i) for i in range(1, 31)]
        forecast1 = pd.DataFrame(index=forecast_dates1, columns=['Forecast'])

        # Use the last 100 days of data for forecasting for first symbol
        last_100_days1 = stock_data1['Close'].tail(100)
        last_100_days_scaled1 = scaler1.transform(np.array(last_100_days1).reshape(-1, 1))

        for i in range(30):
            if selected_model != "LSTM":
                x_forecast1 = last_100_days_scaled1[-100:].reshape(1, -1)
                y_forecast1 = model1.predict(x_forecast1)
            else:
                x_forecast1 = last_100_days_scaled1.reshape(1, -1, 1)
                y_forecast1 = lstm_model1.predict(x_forecast1)
            forecast1.iloc[i] = scaler1.inverse_transform(y_forecast1)[0][0]
            last_100_days_scaled1 = np.append(last_100_days_scaled1, y_forecast1)

        st.subheader(f'30-Day Forecast for {stock_symbol1}')
        st.write(forecast1)

        # Repeat the above process for the second stock symbol
        # Download stock data for second symbol
        stock_data2 = yf.download(stock_symbol2, start=start_date, end=end_date)
        st.subheader(f'Stock Data for {stock_symbol2}')
        st.write(stock_data2.head(50))  # Display first 50 rows

        # Calculate moving averages for second symbol
        stock_data2['MA100'] = calculate_moving_average(stock_data2['Close'], 100)
        stock_data2['MA200'] = calculate_moving_average(stock_data2['Close'], 200)

        # Plot stock data with moving averages for second symbol
        st.subheader(f'Price vs MA100 for {stock_symbol2}')
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=stock_data2.index, y=stock_data2['Close'], mode='lines', name='Close Price'))
        fig4.add_trace(go.Scatter(x=stock_data2.index, y=stock_data2['MA100'], mode='lines', name='MA100'))
        st.plotly_chart(fig4)

        # Plot stock data with moving averages for second symbol
        st.subheader(f'Price vs MA100 vs MA200 for {stock_symbol2}')
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=stock_data2.index, y=stock_data2['Close'], mode='lines', name='Close Price'))
        fig5.add_trace(go.Scatter(x=stock_data2.index, y=stock_data2['MA100'], mode='lines', name='MA100'))
        fig5.add_trace(go.Scatter(x=stock_data2.index, y=stock_data2['MA200'], mode='lines', name='MA200'))
        st.plotly_chart(fig5)

        # Load trained model based on selection for second symbol
        if selected_model == "Neural Network":
            model2 = load_model('KNN_model.keras')
        elif selected_model == "Random Forest":
            model2 = load_model('random_forest_model.keras')
        elif selected_model == "Linear Regression":
            model2 = load_model('linear_regression_model.keras')
        elif selected_model == "LSTM":
            lstm_model2 = load_model('lstm_model.keras')

        # Scale data for second symbol
        scaler2 = MinMaxScaler(feature_range=(0, 1))
        scaled_data2 = scaler2.fit_transform(np.array(stock_data2['Close']).reshape(-1, 1))

        # Prepare data for prediction for second symbol
        x_pred2 = create_dataset(scaled_data2)

        # Predict stock prices for second symbol
        if selected_model != "LSTM":
            y_pred2 = model2.predict(x_pred2)
            y_pred2 = scaler2.inverse_transform(y_pred2)
        else:
            lstm_x_pred2 = create_dataset(scaled_data2)
            lstm_y_pred2 = lstm_model2.predict(lstm_x_pred2)
            lstm_y_pred2 = scaler2.inverse_transform(lstm_y_pred2)

        # Plot original vs predicted prices for second symbol
        st.subheader(f'Original vs Predicted Prices for {stock_symbol2}')
        fig6 = go.Figure()
        fig6.add_trace(go.Scatter(x=stock_data2.index, y=stock_data2['Close'], mode='lines', name='Original Price'))
        if selected_model != "LSTM":
            fig6.add_trace(go.Scatter(x=stock_data2.index[100:], y=y_pred2.flatten(), mode='lines', name='Predicted Price'))
        else:
            fig6.add_trace(go.Scatter(x=stock_data2.index[100:], y=lstm_y_pred2.flatten(), mode='lines', name='LSTM Predicted Price'))
        st.plotly_chart(fig6)

        # Evaluation metrics for second symbol
        y_true2 = stock_data2['Close'].values[100:]
        if selected_model != "LSTM":
            mae2 = mean_absolute_error(y_true2, y_pred2)
            mse2 = mean_squared_error(y_true2, y_pred2)
        else:
            mae2 = mean_absolute_error(y_true2, lstm_y_pred2)
            mse2 = mean_squared_error(y_true2, lstm_y_pred2)

        st.subheader(f'Model Evaluation for {stock_symbol2}')
        st.write(f'Mean Absolute Error (MAE): {mae2:.2f}')
        st.write(f'Mean Squared Error (MSE): {mse2:.2f}')

        # Forecasting for second symbol
        forecast_dates2 = [stock_data2.index[-1] + timedelta(days=i) for i in range(1, 31)]
        forecast2 = pd.DataFrame(index=forecast_dates2, columns=['Forecast'])

        # Use the last 100 days of data for forecasting for second symbol
        last_100_days2 = stock_data2['Close'].tail(100)
        last_100_days_scaled2 = scaler2.transform(np.array(last_100_days2).reshape(-1, 1))

        for i in range(30):
            if selected_model != "LSTM":
                x_forecast2 = last_100_days_scaled2[-100:].reshape(1, -1)
                y_forecast2 = model2.predict(x_forecast2)
            else:
                x_forecast2 = last_100_days_scaled2.reshape(1, -1, 1)
                y_forecast2 = lstm_model2.predict(x_forecast2)
            forecast2.iloc[i] = scaler2.inverse_transform(y_forecast2)[0][0]
            last_100_days_scaled2 = np.append(last_100_days_scaled2, y_forecast2)

        st.subheader(f'30-Day Forecast for {stock_symbol2}')
        st.write(forecast2)

    except Exception as e:
        st.error(f"Error: {e}")

# Streamlit app
def main():
    st.sidebar.title('Stock Comparison App')
    st.sidebar.markdown('Copyright by Rajdeep Sarkar')

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
        st.subheader("Comparison of Two Stocks")
        compare_stocks(stock_symbol1, stock_symbol2, start_date, end_date, selected_model)

if __name__ == '__main__':
    main()
