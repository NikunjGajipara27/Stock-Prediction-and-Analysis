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
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, MACD
import logging
import plotly.express as px
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout


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


def gyaani_baba_prediction(symbol, days=120):
    """Predict future stock prices using a Random Forest model."""
    try:
        data = fetch_stock_data(symbol, datetime.date.today() - pd.DateOffset(years=1), datetime.date.today())
        data['SMA_50'] = SMAIndicator(data['Close']).sma_indicator()
        data['SMA_200'] = SMAIndicator(data['Close']).sma_indicator()
        data['RSI'] = RSIIndicator(data['Close'].values.flatten(), window=14).rsi()
        data['MACD'] = MACD(data['Close'].values.flatten()).macd_diff()
        
        # Drop rows with missing values after creating indicators
        data = data.dropna()
        
        X = data[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_50', 'SMA_200', 'RSI', 'MACD']]
        y = data['Close'].shift(-1).dropna()  # Predict next day's close price
        X = X.iloc[:-1]  # Drop last row to align with y
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
        train_r2 = r2_score(y_train, train_predictions) * 100
        test_r2 = r2_score(y_test, test_predictions) * 100
        
        # Generate future predictions
        last_data = X.iloc[-1:].values  # Ensure it's properly shaped (1D)
        predictions = []
        for _ in range(days):
            pred = model.predict(last_data.reshape(1, -1))[0]
            predictions.append(pred)
            last_data = np.roll(last_data, -1)
            last_data[0, -1] = pred
        
        # Display model performance metrics
        st.write("Model Performance Metrics:")
        st.write("--------------------------------------------------")
        st.write(f"Training R-squared Score: {train_r2:.2f}%")
        st.write(f"Testing R-squared Score: {test_r2:.2f}%")
        st.write(f"Training RMSE: ₹{train_rmse:.2f}")
        st.write(f"Testing RMSE: ₹{test_rmse:.2f}")
        st.write(f"Training Accuracy: {100 - train_rmse:.2f}%")
        
        # Display predictions
        future_dates = pd.date_range(datetime.date.today(), periods=days)
        pred_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': predictions})
        st.write("Predicted Prices:")
        st.write(pred_df)
        st.line_chart(pred_df.set_index('Date'))
        
        return predictions
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
        st.write(f"Time Series Analysis of {symbol}:")
        st.write(data)

        # Use correct column reference for multi-indexed DataFrame
        fig = px.line(data, x=data.index, y=data[('Close', symbol)], title=f"{symbol} Closing Price Over Time")
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Closing Price")
        st.plotly_chart(fig, use_container_width=True)
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