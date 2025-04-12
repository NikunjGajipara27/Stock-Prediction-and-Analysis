import yfinance as yf
from gnews import GNews
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
import pandas as pd
import streamlit as st
import datetime
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, MACD
import logging
import plotly.express as px
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download and setup NLTK for sentiment analysis
try:
    import nltk
    nltk.download('vader_lexicon', quiet=True)
except Exception as e:
    logging.warning(f"Error downloading NLTK data: {e}")

# Set the title of the Streamlit app
st.markdown("<h1 style='text-align: center; color: #0073e6;'>🚀 Stock Analysis Dashboard By Nikunj 🚀</h1>", unsafe_allow_html=True)

# List of NSE stock symbols
stock_symbols = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "WIPRO.NS",
    "ITC.NS", "BAJAJFINSV.NS", "MARUTI.NS", "SBIN.NS", "NESTLEIND.NS",
    "HINDUNILVR.NS", "BRITANNIA.NS", "ULTRACEMCO.NS", "GRASIM.NS",
    "TATAMOTORS.NS", "POWERGRID.NS", "TECHM.NS", "CIPLA.NS", "ZOMATO.NS",
    "ADANIENT.NS","ADANIPOWER.NS","ADANIGREEN.NS","IRFC.NS","RVNL.NS",
    "VOLTAS.NS","TATATECH.NS","TATASTEEL.NS","MRF.NS","SWIGGY.NS","DMART.NS",
    "APOLLOHOSP.NS","BHARTIARTL.NS","AXISBANK.NS","ICICIBANK.NS","JSWSTEEL.NS",
    "AMZN","MSFT","TSLA","META","NVDA","IBM","WMT","GOOG","ORCL","ADBE",
    "INTC","AAPL"
]

# Sidebar selection for options
st.sidebar.header("Select an Option")
option = st.sidebar.selectbox("Choose Analysis", (
    "Overall Market Status",
    "Current Price",
    "Price Between Dates",
    "Stock Comparison",
    "Time Series Analysis",
    "Fundamental Analysis",
    "Prediction (Gyaani Baba)",
    "Technical Analysis"
))


@st.cache_data(ttl=3600)
def fetch_stock_data(symbol, start_date, end_date):
    """Fetch historical stock data for a specific symbol within a date range."""
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        if data.empty:
            st.warning(f"No data available for {symbol} between {start_date} and {end_date}.")
        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return pd.DataFrame()


def fetch_market_data():
    """Retrieve current data for major market indices."""
    indices = {
        "NIFTY": "^NSEI",
        "SENSEX": "^BSESN",
        "NIFTY BANK": "^NSEBANK",
        "Nikkei 225": "^N225",
        "Dow Jones": "^DJI"
    }
    market_data = {}
    for name, symbol in indices.items():
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d")
            if not data.empty:
                close_price = data['Close'].iloc[-1]
                change = close_price - data['Open'].iloc[0]
                percent_change = (change / data['Open'].iloc[0]) * 100
                market_data[name] = {
                    "price": close_price,
                    "change": change,
                    "percent_change": percent_change
                }
        except Exception as e:
            st.error(f"Error fetching data for {name}: {str(e)}")
    return market_data


def display_market_interface(market_data):
    """Display overall market data in a grid layout."""
    cols = st.columns(len(market_data))
    for i, (name, data) in enumerate(market_data.items()):
        with cols[i]:
            st.metric(label=name, value=f"{data['price']:.2f}", delta=f"{data['change']:.2f} ({data['percent_change']:.2f}%)")
    
    # Fetch overall market Intraday Data
    nifty_data = yf.download('^NSEI', period='1d', interval='5m')
    sensex_data = yf.download('^BSESN', period='1d', interval='5m')
    niftybank_data = yf.download('^NSEBANK', period='1d', interval='5m')
    Nikkei225_data = yf.download('^N225', period='1d', interval='5m')
    dowJones_data = yf.download('^DJI', period='1d', interval='5m')
    
    if not nifty_data.empty:
        # Apply a rolling average for better smoothing
        nifty_data['Close_Smooth'] = nifty_data['Close'].rolling(window=3, min_periods=1).mean()

        # Ensure index is in datetime format
        nifty_data.index = pd.to_datetime(nifty_data.index)

        nifty_data.index = nifty_data.index + pd.DateOffset(hours=5, minutes=30)  # Shift by 5:30 hours to match IST

        # Create the figure
        fig = go.Figure()

        # Add trace for closing prices
        fig.add_trace(go.Scatter(
            x=nifty_data.index, 
            y=nifty_data['Close_Smooth'], 
            mode='lines',
            name="NIFTY Close Price",
            line=dict(color='cyan', width=2)
        ))

        # Update layout for better visualization
        fig.update_layout(
            title='NIFTY Intraday Chart',
            xaxis_title='Time',
            yaxis_title='Price',
            template='plotly_dark',
            xaxis=dict(
                showgrid=True,
                tickformat="%H:%M",  # Display time in HH:MM format
                tickangle=-45,  # Rotate labels for better readability
                dtick=30 * 60 * 1000  # Set interval to 30 minutes (milliseconds)
                ),
            yaxis=dict(showgrid=True),
            height=500,
            width=900
        )

        # Display chart
        st.plotly_chart(fig)

    if not sensex_data.empty:
        # Apply a rolling average for better smoothing
        sensex_data['Close_Smooth'] = sensex_data['Close'].rolling(window=3, min_periods=1).mean()

        # Ensure index is in datetime format
        sensex_data.index = pd.to_datetime(sensex_data.index)

        sensex_data.index = sensex_data.index + pd.DateOffset(hours=5, minutes=30)  # Shift by 5:30 hours to match IST

        # Create the figure
        fig = go.Figure()

        # Add trace for closing prices
        fig.add_trace(go.Scatter(
            x=sensex_data.index, 
            y=sensex_data['Close_Smooth'], 
            mode='lines',
            name="SENSEX Close Price",
            line=dict(color='cyan', width=2)
        ))

        # Update layout for better visualization
        fig.update_layout(
            title='SENSEX Intraday Chart',
            xaxis_title='Time',
            yaxis_title='Price',
            template='plotly_dark',
            xaxis=dict(
                showgrid=True,
                tickformat="%H:%M",  # Display time in HH:MM format
                tickangle=-45,  # Rotate labels for better readability
                dtick=30 * 60 * 1000  # Set interval to 30 minutes (milliseconds)
                ),
            yaxis=dict(showgrid=True),
            height=500,
            width=900
        )

        # Display chart
        st.plotly_chart(fig)

    if not niftybank_data.empty:
        # Apply a rolling average for better smoothing
        niftybank_data['Close_Smooth'] = niftybank_data['Close'].rolling(window=3, min_periods=1).mean()

        # Ensure index is in datetime format
        niftybank_data.index = pd.to_datetime(niftybank_data.index)

        niftybank_data.index = niftybank_data.index + pd.DateOffset(hours=5, minutes=30)  # Shift by 5:30 hours to match IST

        # Create the figure
        fig = go.Figure()

        # Add trace for closing prices
        fig.add_trace(go.Scatter(
            x=niftybank_data.index, 
            y=niftybank_data['Close_Smooth'], 
            mode='lines',
            name="NIFTY BANK Close Price",
            line=dict(color='cyan', width=2)
        ))

        # Update layout for better visualization
        fig.update_layout(
            title='NIFTY BANK Intraday Chart',
            xaxis_title='Time',
            yaxis_title='Price',
            template='plotly_dark',
            xaxis=dict(
                showgrid=True,
                tickformat="%H:%M",  # Display time in HH:MM format
                tickangle=-45,  # Rotate labels for better readability
                dtick=30 * 60 * 1000  # Set interval to 30 minutes (milliseconds)
                ),
            yaxis=dict(showgrid=True),
            height=500,
            width=900
        )

        # Display chart
        st.plotly_chart(fig)

    if not Nikkei225_data.empty:
        # Apply a rolling average for better smoothing
        Nikkei225_data['Close_Smooth'] = Nikkei225_data['Close'].rolling(window=3, min_periods=1).mean()

        # Ensure index is in datetime format
        Nikkei225_data.index = pd.to_datetime(Nikkei225_data.index)

        Nikkei225_data.index = Nikkei225_data.index + pd.DateOffset(hours=9, minutes=00)  # Shift by 9:00 hours to match IST

        # Create the figure
        fig = go.Figure()

        # Add trace for closing prices
        fig.add_trace(go.Scatter(
            x=Nikkei225_data.index, 
            y=Nikkei225_data['Close_Smooth'], 
            mode='lines',
            name="Nikkei 225 Close Price",
            line=dict(color='cyan', width=2)
        ))

        # Update layout for better visualization
        fig.update_layout(
            title='Nikkei 225 Intraday Chart',
            xaxis_title='Time',
            yaxis_title='Price',
            template='plotly_dark',
            xaxis=dict(
                showgrid=True,
                tickformat="%H:%M",  # Display time in HH:MM format
                tickangle=-45,  # Rotate labels for better readability
                dtick=30 * 60 * 1000  # Set interval to 30 minutes (milliseconds)
                ),
            yaxis=dict(showgrid=True),
            height=500,
            width=900
        )

        # Display chart
        st.plotly_chart(fig)

    if not dowJones_data.empty:
        # Apply a rolling average for better smoothing
        dowJones_data['Close_Smooth'] = dowJones_data['Close'].rolling(window=3, min_periods=1).mean()

        # Ensure index is in datetime format
        dowJones_data.index = pd.to_datetime(dowJones_data.index)

        dowJones_data.index = dowJones_data.index - pd.DateOffset(hours=5, minutes=00)  # Reduce by 5:00 hours to match IST

        # Create the figure
        fig = go.Figure()

        # Add trace for closing prices
        fig.add_trace(go.Scatter(
            x=dowJones_data.index, 
            y=dowJones_data['Close_Smooth'], 
            mode='lines',
            name="Dow Jones Close Price",
            line=dict(color='cyan', width=2)
        ))

        # Update layout for better visualization
        fig.update_layout(
            title='Dow Jones Intraday Chart',
            xaxis_title='Time',
            yaxis_title='Price',
            template='plotly_dark',
            xaxis=dict(
                showgrid=True,
                tickformat="%H:%M",  # Display time in HH:MM format
                tickangle=-45,  # Rotate labels for better readability
                dtick=30 * 60 * 1000  # Set interval to 30 minutes (milliseconds)
                ),
            yaxis=dict(showgrid=True),
            height=500,
            width=900
        )

        # Display chart
        st.plotly_chart(fig)


def fetch_news_sentiment(symbol):
    """Fetch news and analyze sentiment for a given stock symbol."""
    try:
        gnews = GNews(language='en', country='IN', max_results=10)
        news = gnews.get_news(symbol)
        sentiment_analyzer = SentimentIntensityAnalyzer()
        sentiments = []
        
        st.write("\nLatest News and Sentiments:")
        st.write("--------------------------------------------------")
        
        for article in news:
            title = article['title']
            sentiment_score = sentiment_analyzer.polarity_scores(title)['compound']
            sentiments.append(sentiment_score)
            sentiment_label = "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"
            st.write(f"Title: {title}\nSentiment: {sentiment_label} (Score: {sentiment_score:.2f})\n")
        
        avg_sentiment = np.mean(sentiments) if sentiments else 0
        overall_sentiment = "Positive" if avg_sentiment > 0 else "Negative" if avg_sentiment < 0 else "Neutral"
        
        st.write("--------------------------------------------------")
        st.write(f"Overall Sentiment: {overall_sentiment} (Score: {avg_sentiment:.2f})")
    except Exception as e:
        st.error(f"Error in sentiment analysis: {str(e)}")


def gyaani_baba_prediction(symbol, days=30):
    """Predict future stock prices using an enhanced LSTM model with calibration."""
    try:
        # Fetch stock data for the past 5 years
        start_date = datetime.date.today() - pd.DateOffset(years=5)
        end_date = datetime.date.today()
        data = fetch_stock_data(symbol, start_date, end_date)

        if data.empty:
            st.error("No data found for the given stock symbol.")
            return None

        # Prepare data for model
        df = pd.DataFrame()
        df['Open'] = data['Open']
        df['High'] = data['High']
        df['Low'] = data['Low']
        df['Close'] = data['Close']
        df['Volume'] = data['Volume']

        # Add technical indicators
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
        df['MACD'] = MACD(df['Close']).macd_diff()

        # Add price momentum features
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_5d'] = df['Close'].pct_change(5)

        # Remove NaN values
        df = df.dropna()

        # Normalize data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df)

        # Create sequences for LSTM
        sequence_length = 120  # Increased sequence length
        X, y = [], []
        close_index = list(df.columns).index('Close')
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i])
            y.append(scaled_data[i, close_index])

        X, y = np.array(X), np.array(y)

        # Split into train and test sets
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Define LSTM model
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.3),
            LSTM(64, return_sequences=False),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1)
        ])

        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)  # Reduced learning rate
        model.compile(optimizer=optimizer, loss='mean_squared_error')

        # Train model with early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        history = model.fit(
            X_train, y_train,
            validation_split=0.2,  # Increased validation split
            epochs=30,  # Increased epochs
            batch_size=32,
            callbacks=[early_stopping],
            verbose=1
        )

        # Evaluate model
        train_pred = model.predict(X_train).flatten()
        test_pred = model.predict(X_test).flatten()

        # Inverse transform predictions
        close_scaler = MinMaxScaler(feature_range=(0, 1))
        close_scaler.fit(df[['Close']])
        train_pred_inv = close_scaler.inverse_transform(train_pred.reshape(-1, 1)).flatten()
        test_pred_inv = close_scaler.inverse_transform(test_pred.reshape(-1, 1)).flatten()
        y_train_inv = close_scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
        y_test_inv = close_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        # Calculate metrics
        train_mae = mean_absolute_error(y_train_inv, train_pred_inv)
        test_mae = mean_absolute_error(y_test_inv, test_pred_inv)
        train_rmse = np.sqrt(mean_squared_error(y_train_inv, train_pred_inv))
        test_rmse = np.sqrt(mean_squared_error(y_test_inv, test_pred_inv))
        train_r2 = r2_score(y_train_inv, train_pred_inv)
        test_r2 = r2_score(y_test_inv, test_pred_inv)

        # Display metrics
        print("### Model Performance Metrics")
        print("#### Training Data")
        print(f"Mean Absolute Error (MAE): {train_mae:.4f}")
        print(f"Root Mean Squared Error (RMSE): {train_rmse:.4f}")
        print(f"R-squared (R²): {train_r2:.4f}")

        print("#### Testing Data")
        print(f"Mean Absolute Error (MAE): {test_mae:.4f}")
        print(f"Root Mean Squared Error (RMSE): {test_rmse:.4f}")
        print(f"R-squared (R²): {test_r2:.4f}")

        # Predict future prices
        future_predictions = []
        last_sequence = scaled_data[-sequence_length:]
        current_batch = last_sequence.reshape(1, sequence_length, scaled_data.shape[1])

        for _ in range(days):
            next_pred = model.predict(current_batch, verbose=0)[0][0]
            future_predictions.append(next_pred)
            new_point = current_batch[0][-1].copy()
            new_point[close_index] = next_pred
            current_batch = np.append(current_batch[:, 1:, :],
                                    [new_point.reshape(1, scaled_data.shape[1])],
                                    axis=1)

        # Inverse transform predictions
        future_predictions = np.array(future_predictions).reshape(-1, 1)
        predicted_prices = close_scaler.inverse_transform(future_predictions).flatten()

        # Calibration with current market price
        ticker = yf.Ticker(symbol)
        currency_symbol = "₹" if symbol.endswith(".NS") else "$"
        current_market_price = ticker.info.get('currentPrice', df['Close'].iloc[-1])
        calibration_factor = current_market_price / predicted_prices[0]
        calibrated_predictions = predicted_prices * calibration_factor

        # Display predictions
        st.write("### 🔮 Future Stock Price Predictions")
        df_predictions = pd.DataFrame({
            "Day": list(range(1, days + 1)),
            f"Calibrated Predicted Price ({currency_symbol})": calibrated_predictions
        })
        st.dataframe(df_predictions)

        # Plot predictions
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index[-60:], y=df['Close'][-60:], mode='lines', name='Historical'))
        fig.add_trace(go.Scatter(x=pd.date_range(start=df.index[-1], periods=days+1, freq='B')[1:],
                               y=calibrated_predictions, mode='lines', name='Forecast', line=dict(dash='dash')))
        fig.update_layout(title=f"{symbol} {days}-Day Forecast",
                         xaxis_title="Date",
                         yaxis_title="Price",
                         template="plotly_dark")
        # st.plotly_chart(fig)

        return calibrated_predictions

    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None

if option == "Overall Market Status":
    market_data = fetch_market_data()
    display_market_interface(market_data)

elif option == "Current Price":
    symbol = st.sidebar.selectbox("Select Stock", stock_symbols)
    ticker = yf.Ticker(symbol)
    currency_symbol = "₹" if symbol.endswith(".NS") else "$"  # Check if it's an Indian stock
    st.write(f"Current Price of {symbol} : {currency_symbol}{ticker.info.get('currentPrice', 'N/A')}")
    fetch_news_sentiment(symbol)

elif option == "Price Between Dates":
    symbol = st.sidebar.selectbox("Select Stock", stock_symbols)
    start_date = st.sidebar.date_input("Start Date")
    end_date = st.sidebar.date_input("End Date")
    
    data = fetch_stock_data(symbol, start_date, end_date)
    
    if not data.empty:
        st.write(f"Price of {symbol} between {start_date} and {end_date}:")
        st.write(data)

        # Ensure correct column reference for multi-indexed DataFrame
        if ('Close', symbol) in data.columns:
            fig = px.line(data, x=data.index, y=data[('Close', symbol)], 
                          title=f"{symbol} Closing Price Over Time")
            fig.update_xaxes(title_text="Date")
            fig.update_yaxes(title_text="Closing Price")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"Column ('Close', {symbol}) not found in data. Please check the structure.")
    else:
        st.error(f"No time series data available for {symbol}.")

elif option == "Stock Comparison":
    symbols = st.sidebar.multiselect("Select Stocks for Comparison", stock_symbols)
    start_date = st.sidebar.date_input("Start Date")
    end_date = st.sidebar.date_input("End Date")
    
    data = {symbol: fetch_stock_data(symbol, start_date, end_date) for symbol in symbols}

    if all(not df.empty for df in data.values()):
        st.write("Stock Comparison:")

        for symbol, df in data.items():
            st.write(f"### {symbol} Stock Data")
            st.write(df)

            # Ensure correct column reference for multi-indexed DataFrame
            if ('Close', symbol) in df.columns:
                fig = px.line(df, x=df.index, y=df[('Close', symbol)], 
                              title=f"{symbol} Closing Price Over Time")
                fig.update_xaxes(title_text="Date")
                fig.update_yaxes(title_text="Closing Price")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"Column ('Close', {symbol}) not found in {symbol}'s data. Please check the structure.")
    else:
        st.error("No data available for the selected stocks and date range.")
            

elif option == "Time Series Analysis":
    symbol = st.sidebar.selectbox("Select Stock", stock_symbols)
    data = fetch_stock_data(symbol, datetime.date.today() - pd.DateOffset(years=5), datetime.date.today())

    if not data.empty:
        st.write(f"### Time Series Analysis of {symbol}")
        st.write("Analyzing 5 years of historical data...")

        # Use the correct multi-index column for 'Close'
        close_col = ('Close', symbol) if ('Close', symbol) in data.columns else 'Close'  # Fallback to 'Close' if no multi-index
        close_prices = data[close_col]

        # 1. Basic Plot of Closing Prices
        st.write("#### Historical Closing Prices")
        # Create a DataFrame with explicit columns to avoid length mismatch
        plot_data = pd.DataFrame({
            'Date': data.index,
            'Close': data[close_col]
        })
        fig = px.line(plot_data, x='Date', y='Close', title=f"{symbol} Closing Price Over Time")
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Closing Price")
        st.plotly_chart(fig, use_container_width=True)

        # 2. Trend Decomposition
        st.write("#### Trend Decomposition")
        try:
            decomposition = seasonal_decompose(close_prices, model='additive', period=252)  # 252 trading days ~ 1 year
            trend = decomposition.trend
            seasonal = decomposition.seasonal
            residual = decomposition.resid

            fig_decomp = go.Figure()
            fig_decomp.add_trace(go.Scatter(x=close_prices.index, y=close_prices, mode='lines', name='Original'))
            fig_decomp.add_trace(go.Scatter(x=trend.index, y=trend, mode='lines', name='Trend'))
            fig_decomp.add_trace(go.Scatter(x=seasonal.index, y=seasonal, mode='lines', name='Seasonal'))
            fig_decomp.add_trace(go.Scatter(x=residual.index, y=residual, mode='lines', name='Residual'))
            fig_decomp.update_layout(
                title=f"{symbol} Time Series Decomposition",
                xaxis_title="Date",
                yaxis_title="Value",
                template="plotly_dark",
                height=600
            )
            st.plotly_chart(fig_decomp, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not decompose series: {str(e)}")

        # 3. Stationarity Test (ADF)
        st.write("#### Stationarity Test (Augmented Dickey-Fuller)")
        adf_result = adfuller(close_prices.dropna())
        st.write(f"ADF Statistic: {adf_result[0]:.4f}")
        st.write(f"p-value: {adf_result[1]:.4f}")
        st.write("Interpretation: " + ("Stationary" if adf_result[1] < 0.05 else "Non-Stationary"))

        # 4. Autocorrelation Analysis
        st.write("#### Autocorrelation Function (ACF)")
        acf_values = acf(close_prices.dropna(), nlags=40)
        fig_acf = go.Figure()
        fig_acf.add_trace(go.Bar(x=np.arange(len(acf_values)), y=acf_values, name='ACF'))
        fig_acf.update_layout(
            title=f"{symbol} Autocorrelation",
            xaxis_title="Lag",
            yaxis_title="Autocorrelation",
            template="plotly_dark",
            height=400
        )
        st.plotly_chart(fig_acf, use_container_width=True)

        # 5. LSTM Forecasting
        st.write("#### LSTM Forecast (30 Days Ahead)")
        try:
            # Prepare data for LSTM
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(close_prices.values.reshape(-1, 1))

            # Create sequences
            sequence_length = 60
            X, y = [], []
            for i in range(sequence_length, len(scaled_data)):
                X.append(scaled_data[i-sequence_length:i])
                y.append(scaled_data[i])
            X, y = np.array(X), np.array(y)

            # Split into train and test sets
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            # Define LSTM model
            model = Sequential([
                LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
                Dropout(0.2),
                LSTM(units=50, return_sequences=False),
                Dropout(0.2),
                Dense(units=25),
                Dense(units=1)
            ])

            # Compile and train
            model.compile(optimizer='adam', loss='mean_squared_error')
            history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=0)

            # Model Performance Metrics
            test_pred = model.predict(X_test).flatten()
            train_pred = model.predict(X_train).flatten()
        
            # Define close_scaler before using it
            close_scaler = MinMaxScaler(feature_range=(0, 1))
            close_scaler.fit(close_prices.values.reshape(-1, 1))
        
            # Inverse transform for metrics
            train_pred_inv = close_scaler.inverse_transform(train_pred.reshape(-1, 1)).flatten()
            test_pred_inv = close_scaler.inverse_transform(test_pred.reshape(-1, 1)).flatten()
            y_train_inv = close_scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
            y_test_inv = close_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

            # Now calculate real-world metrics
            train_mae = np.mean(np.abs(y_train_inv - train_pred_inv))
            test_mae = np.mean(np.abs(y_test_inv - test_pred_inv))
            train_rmse = np.sqrt(np.mean((y_train_inv - train_pred_inv) ** 2))
            test_rmse = np.sqrt(np.mean((y_test_inv - test_pred_inv) ** 2))
            

            # Evaluate model on training data
            y_train_pred = model.predict(X_train, verbose=0)
            train_r2 = r2_score(y_train, y_train_pred)
            train_mape = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100

            # Evaluate model on testing data
            y_test_pred = model.predict(X_test, verbose=0)
            test_r2 = r2_score(y_test, y_test_pred)
            test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100

           # Display performance metrics
            st.write("### Model Performance Metrics")
            st.write("#### Training Data")
            st.write(f"Mean Squared Error (MSE): {train_mae:.4f}")
            st.write(f"Root Mean Squared Error (RMSE): {train_rmse:.4f}")
            st.write(f"Mean Absolute Error (MAE): {train_mae:.4f}")
            st.write(f"R-squared (R²): {train_r2:.4f}")
            st.write(f"Mean Absolute Percentage Error (MAPE): {train_mape:.2f}%")

            st.write("#### Testing Data")
            st.write(f"Mean Squared Error (MSE): {test_mae:.4f}")
            st.write(f"Root Mean Squared Error (RMSE): {test_rmse:.4f}")
            st.write(f"Mean Absolute Error (MAE): {test_mae:.4f}")
            st.write(f"R-squared (R²): {test_r2:.4f}")
            st.write(f"Mean Absolute Percentage Error (MAPE): {test_mape:.2f}%")
            
            # Forecast 30 days ahead
            last_sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)
            future_predictions = []
            for _ in range(30):
                next_pred = model.predict(last_sequence, verbose=0)
                future_predictions.append(next_pred[0, 0])
                last_sequence = np.roll(last_sequence, -1, axis=1)
                last_sequence[0, -1, 0] = next_pred[0, 0]

            # Inverse transform predictions
            future_predictions = np.array(future_predictions).reshape(-1, 1)
            predicted_prices = scaler.inverse_transform(future_predictions)

            # Create forecast index
            last_date = close_prices.index[-1]
            forecast_index = pd.date_range(start=last_date, periods=31, freq='B')[1:]  # Business days

            # Plot historical data and forecast
            fig_forecast = go.Figure()
            fig_forecast.add_trace(go.Scatter(x=close_prices.index[-100:], y=close_prices[-100:], mode='lines', name='Historical'))
            fig_forecast.add_trace(go.Scatter(x=forecast_index, y=predicted_prices.flatten(), mode='lines', name='LSTM Forecast', line=dict(dash='dash')))
            fig_forecast.update_layout(
                title=f"{symbol} 30-Day LSTM Forecast",
                xaxis_title="Date",
                yaxis_title="Price",
                template="plotly_dark",
                height=500
            )
            st.plotly_chart(fig_forecast, use_container_width=True)

            # Display forecast values
            forecast_df = pd.DataFrame({'Date': forecast_index, 'Forecasted Price': predicted_prices.flatten()})
            st.write(forecast_df)

        except Exception as e:
            st.warning(f"Could not fit LSTM model: {str(e)}")

    else:
        st.error(f"No time series data available for {symbol}.")

elif option == "Fundamental Analysis":
    symbol = st.sidebar.selectbox("Select Stock", stock_symbols)
    currency_symbol = "₹" if symbol.endswith(".NS") else "$"  # Check if it's an Indian stock
    ticker = yf.Ticker(symbol)
    info = ticker.info
    st.write(f"Fundamental Analysis of {symbol}:")
    st.write("--------------------------------------------------")
    st.write(f"Market Cap: {currency_symbol}{info.get('marketCap', 'N/A')}")
    st.write(f"PE Ratio: {info.get('trailingPE', 'N/A')}")
    st.write(f"Dividend Yield: {info.get('dividendYield', 'N/A')}")
    st.write(f"EPS: {currency_symbol}{info.get('trailingEps', 'N/A')}")
    st.write(f"52-Week High: {currency_symbol}{info.get('fiftyTwoWeekHigh', 'N/A')}")
    st.write(f"52-Week Low: {currency_symbol}{info.get('fiftyTwoWeekLow', 'N/A')}")

elif option == "Prediction (Gyaani Baba)":
    symbol = st.sidebar.selectbox("Select Stock for Prediction", stock_symbols)
    days = st.sidebar.slider("Days to Predict", 1, 120)
    predictions = gyaani_baba_prediction(symbol, days)
    fetch_news_sentiment(symbol)

elif option == "Technical Analysis":
    symbol = st.sidebar.selectbox("Select Stock", stock_symbols)
    data = fetch_stock_data(symbol, datetime.date.today() - pd.DateOffset(years=3), datetime.date.today())
    if not data.empty:
        st.write(f"Technical Analysis of {symbol}:")
        st.write("--------------------------------------------------")
        close_prices = data['Close']
        if isinstance(close_prices, pd.DataFrame):
            close_prices = close_prices.squeeze()  # Convert to Series if necessary

        sma_50 = SMAIndicator(close_prices, window=50).sma_indicator()
        sma_200 = SMAIndicator(close_prices, window=200).sma_indicator()
        rsi = RSIIndicator(close_prices, window=14).rsi()
        macd = MACD(close_prices).macd_diff()

        # Display results
        st.write("SMA (50):")
        st.write(sma_50)
        st.write("SMA (200):")
        st.write(sma_200)
        st.write("RSI:")
        st.write(rsi)
        st.write("MACD:")
        st.write(macd)

        # Plot charts
        fig = px.line(data, x=data.index, y=data[('Close', symbol)], title=f"{symbol} Closing Price", labels={"Close": "Price", "index": "Date"})
        fig.update_traces(line=dict(color="cyan"))  # Set line color
        fig.update_layout(template="plotly_dark")  # Optional: Dark theme
        st.plotly_chart(fig)
        st.markdown("50-day Simple Moving Average (SMA)")
        st.line_chart(sma_50)
        st.markdown("200-day Simple Moving Average (SMA)")
        st.line_chart(sma_200)
        st.markdown("Relative Strength Index (RSI)")
        st.line_chart(rsi)
        st.markdown("Moving Average Convergence Divergence (MACD)")
        st.line_chart(macd)