import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import plotly.graph_objects as go
import numpy as np
import seaborn as sns
import plotly.figure_factory as ff
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose

def main():
    st.title("Quantum Quotient Analytics")

    # Sidebar
    st.sidebar.title("Options")
    symbol = st.sidebar.text_input("Enter Stock Symbol", value='AAPL', max_chars=5)
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime('2021-01-01'))
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime('today'))

    st.subheader("Data Table")
    # Fetch stock data
    @st.cache_resource
    def load_data(symbol, start_date, end_date):
        try:
            data = yf.download(symbol, start=start_date, end=end_date)
            data.reset_index(inplace=True)
            return data
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return pd.DataFrame()

    data = load_data(symbol, start_date, end_date)

    if data.empty:
        st.error("No data available for the selected symbol and date range. Please try again.")
        return

    # Calculate VWAP
    data['VWAP'] = (data['Volume'] * (data['High'] + data['Low'] + data['Close']) / 3).cumsum() / data['Volume'].cumsum()

    st.write(data)

    # Additional information about the stock
    st.subheader("Additional Information")
    ticker = yf.Ticker(symbol)
    st.write(f"**Company Name:** {ticker.info.get('longName', 'N/A')}")
    st.write(f"**Industry:** {ticker.info.get('industry', 'N/A')}")
    st.write(f"**Exchange:** {ticker.info.get('exchange', 'N/A')}")
    st.write(f"**Website:** {ticker.info.get('website', 'N/A')}")
    st.write(f"**Market Cap:** {ticker.info.get('marketCap', 'N/A')}")
    st.write(f"**PE Ratio:** {ticker.info.get('forwardPE', 'N/A')}")

    # Check if 'eps' key exists before accessing it
    if 'eps' in ticker.info:
        st.write(f"**EPS:** {ticker.info['eps']}")
    else:
        st.write("**EPS:** N/A")

    # Time Series Plots
    st.subheader("Closing Price Over Time")
    fig_close = go.Figure(data=go.Scatter(x=data['Date'], y=data['Close'], mode='lines'))
    fig_close.update_layout(title=f"Closing Price of {symbol}")
    st.plotly_chart(fig_close, use_container_width=True)

    st.subheader("Volume Traded Over Time")
    fig_volume = go.Figure(data=go.Bar(x=data['Date'], y=data['Volume']))
    fig_volume.update_layout(title=f"Volume Traded of {symbol}")
    st.plotly_chart(fig_volume, use_container_width=True)

    st.subheader("Opening vs Closing Prices Over Time")
    fig_open_close = go.Figure()
    fig_open_close.add_trace(go.Scatter(x=data['Date'], y=data['Open'], mode='lines', name='Open'))
    fig_open_close.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close'))
    fig_open_close.update_layout(title='Opening vs Closing Prices Over Time')
    st.plotly_chart(fig_open_close, use_container_width=True)

    # Time series decompostion
    st.subheader("Time Series Decomposition")
    decomposition = seasonal_decompose(data['Close'].dropna(), model='multiplicative', period=30)
    fig_decomp = go.Figure()

    fig_decomp.add_trace(go.Scatter(x=data['Date'], y=decomposition.trend, mode='lines', name='Trend'))
    fig_decomp.add_trace(go.Scatter(x=data['Date'], y=decomposition.seasonal, mode='lines', name='Seasonal'))
    fig_decomp.add_trace(go.Scatter(x=data['Date'], y=decomposition.resid, mode='lines', name='Residual'))

    fig_decomp.update_layout(title='Time Series Decomposition', height=600)
    st.plotly_chart(fig_decomp, use_container_width=True)

    # Candlestick Chart
    st.subheader("OHLC Chart")
    fig_candlestick = go.Figure(data=[go.Candlestick(x=data['Date'],
                                                    open=data['Open'],
                                                    high=data['High'],
                                                    low=data['Low'],
                                                    close=data['Close'])])
    fig_candlestick.update_layout(title='OHLC Chart')
    st.plotly_chart(fig_candlestick, use_container_width=True)

    # VWAP
    st.subheader("Volume Weighted Average Price (VWAP) Over Time")
    fig_vwap = go.Figure()
    fig_vwap.add_trace(go.Scatter(x=data['Date'], y=data['VWAP'], mode='lines', name='VWAP'))
    fig_vwap.update_layout(title='Volume Weighted Average Price (VWAP) Over Time')
    st.plotly_chart(fig_vwap, use_container_width=True)

    # Trade Analysis
    st.subheader("Number of Trades Over Time")
    fig_trades = go.Figure()
    fig_trades.add_trace(go.Bar(x=data['Date'], y=data['Volume'], name='Trades'))
    fig_trades.update_layout(title='Number of Trades Over Time')
    st.plotly_chart(fig_trades, use_container_width=True)

    # Comparative Analysis
    symbols_list = ['AAPL', 'MSFT', 'GOOGL']  # Add more symbols as needed
    st.subheader("Comparative Closing Prices")
    fig_compare = go.Figure()
    for symbol in symbols_list:
        compare_data = load_data(symbol, start_date, end_date)
        fig_compare.add_trace(go.Scatter(x=compare_data['Date'], y=compare_data['Close'], mode='lines', name=symbol))
    fig_compare.update_layout(title='Comparative Closing Prices')
    st.plotly_chart(fig_compare, use_container_width=True)

    # Filter numeric columns
    numeric_data = data.select_dtypes(include=[float, int]).columns.tolist()
    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    corr_matrix = data[numeric_data].corr()
    fig_heatmap = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu')
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # Filter numeric columns
    numeric_d = data.select_dtypes(include=[float, int])
    # Pair Plot
    st.subheader("Pair Plot")
    # Select fewer important features to avoid congestion
    selected_columns = numeric_d[['Open', 'High', 'Low', 'Close', 'Volume']]
    fig_pairplot = px.scatter_matrix(selected_columns, dimensions=selected_columns.columns, title="Pair Plot", height=800, width=800)
    st.plotly_chart(fig_pairplot, use_container_width=True)

    # Histogram of Returns
    st.subheader("Histogram of Returns")
    if 'Close' in numeric_d.columns:
        numeric_d['Returns'] = numeric_d['Close'].pct_change()
        fig_histogram = px.histogram(numeric_d, x='Returns', nbins=50, title="Histogram of Returns")
        st.plotly_chart(fig_histogram, use_container_width=True)
    else:
        st.error("No 'Close' column found in the data to calculate returns.")

    # MACD Indicator
    def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
        data['Short_EMA'] = data['Close'].ewm(span=short_window, adjust=False).mean()
        data['Long_EMA'] = data['Close'].ewm(span=long_window, adjust=False).mean()
        data['MACD'] = data['Short_EMA'] - data['Long_EMA']
        data['Signal_Line'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()
        return data

    data = calculate_macd(data)

    st.subheader("MACD Indicator")
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=data['Date'], y=data['MACD'], mode='lines', name='MACD'))
    fig_macd.add_trace(go.Scatter(x=data['Date'], y=data['Signal_Line'], mode='lines', name='Signal Line'))
    fig_macd.update_layout(title='MACD Indicator')
    st.plotly_chart(fig_macd, use_container_width=True)

    # Cumulative return plot
    st.subheader("Cumulative Return Over Time")
    # Compute Daily Return and Cumulative Return
    data['Daily Return'] = data['Close'].pct_change()
    data['Cumulative Return'] = (1 + data['Daily Return']).cumprod()
    fig_cum_return = px.line(data, x='Date', y='Cumulative Return', title='Cumulative Return Over Time')
    st.plotly_chart(fig_cum_return, use_container_width=True)


    # Relative performance comparison
    st.subheader("Relative Performance Comparison")
    symbols_list = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TCS']  
    fig_rel_performance = go.Figure()
    for symbol in symbols_list:
        compare_data = load_data(symbol, start_date, end_date)
        compare_data['Cumulative Return'] = (1 + compare_data['Close'].pct_change()).cumprod()
        fig_rel_performance.add_trace(go.Scatter(x=compare_data['Date'], y=compare_data['Cumulative Return'], mode='lines', name=symbol))
    fig_rel_performance.update_layout(title='Relative Performance Comparison')
    st.plotly_chart(fig_rel_performance, use_container_width=True)

    # Annualized Volatility
    st.subheader("Annualized Volatility")
    data['Daily Return'] = data['Close'].pct_change()
    volatility = data['Daily Return'].rolling(window=252).std() * np.sqrt(252)
    fig_volatility = go.Figure(data=go.Scatter(x=data['Date'], y=volatility, mode='lines', name='Annualized Volatility'))
    fig_volatility.update_layout(title='Annualized Volatility Over Time')
    st.plotly_chart(fig_volatility, use_container_width=True)

    # RSI Chart
    def calculate_rsi(data, window=14):
        delta = data['Close'].diff(1)
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        rs = avg_gain / avg_loss
        data['RSI'] = 100 - (100 / (1 + rs))
        return data

    data = calculate_rsi(data)

    st.subheader("Relative Strength Index (RSI)")
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=data['Date'], y=data['RSI'], mode='lines', name='RSI'))
    fig_rsi.update_layout(title='Relative Strength Index (RSI)', yaxis=dict(range=[0, 100]))
    st.plotly_chart(fig_rsi, use_container_width=True)

    # Bollinger Bands
    def calculate_bollinger_bands(data, window=20):
        data['MA20'] = data['Close'].rolling(window=window).mean()
        data['STD20'] = data['Close'].rolling(window=window).std()
        data['Upper_Band'] = data['MA20'] + (data['STD20'] * 2)
        data['Lower_Band'] = data['MA20'] - (data['STD20'] * 2)
        return data

    data = calculate_bollinger_bands(data)

    st.subheader("Bollinger Bands")
    fig_bollinger = go.Figure()
    fig_bollinger.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close'))
    fig_bollinger.add_trace(go.Scatter(x=data['Date'], y=data['Upper_Band'], mode='lines', name='Upper Band'))
    fig_bollinger.add_trace(go.Scatter(x=data['Date'], y=data['Lower_Band'], mode='lines', name='Lower Band'))
    fig_bollinger.update_layout(title='Bollinger Bands')
    st.plotly_chart(fig_bollinger, use_container_width=True)

    # On-Balance Volume (OBV)
    def calculate_obv(data):
        data['OBV'] = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()
        return data

    data = calculate_obv(data)

    st.subheader("On-Balance Volume (OBV)")
    fig_obv = go.Figure()
    fig_obv.add_trace(go.Scatter(x=data['Date'], y=data['OBV'], mode='lines', name='OBV'))
    fig_obv.update_layout(title='On-Balance Volume (OBV)')
    st.plotly_chart(fig_obv, use_container_width=True)

    # Simple Moving Average (SMA)
    def calculate_sma(data, window):
        data[f'SMA_{window}'] = data['Close'].rolling(window=window).mean()
        return data

    data = calculate_sma(data, window=50)
    data = calculate_sma(data, window=200)

    st.subheader("Simple Moving Average (SMA)")
    fig_sma = go.Figure()
    fig_sma.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close'))
    fig_sma.add_trace(go.Scatter(x=data['Date'], y=data['SMA_50'], mode='lines', name='SMA 50'))
    fig_sma.add_trace(go.Scatter(x=data['Date'], y=data['SMA_200'], mode='lines', name='SMA 200'))
    fig_sma.update_layout(title='Simple Moving Average (SMA)')
    st.plotly_chart(fig_sma, use_container_width=True)

    # Exponential Moving Average (EMA)
    def calculate_ema(data, window):
        data[f'EMA_{window}'] = data['Close'].ewm(span=window, adjust=False).mean()
        return data

    data = calculate_ema(data, window=50)
    data = calculate_ema(data, window=200)

    st.subheader("Exponential Moving Average (EMA)")
    fig_ema = go.Figure()
    fig_ema.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close'))
    fig_ema.add_trace(go.Scatter(x=data['Date'], y=data['EMA_50'], mode='lines', name='EMA 50'))
    fig_ema.add_trace(go.Scatter(x=data['Date'], y=data['EMA_200'], mode='lines', name='EMA 200'))
    fig_ema.update_layout(title='Exponential Moving Average (EMA)')
    st.plotly_chart(fig_ema, use_container_width=True)

    # Average True Range (ATR)
    def calculate_atr(data, window=14):
        data['High-Low'] = data['High'] - data['Low']
        data['High-PrevClose'] = np.abs(data['High'] - data['Close'].shift(1))
        data['Low-PrevClose'] = np.abs(data['Low'] - data['Close'].shift(1))
        data['TrueRange'] = data[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
        data['ATR'] = data['TrueRange'].rolling(window=window).mean()
        return data

    data = calculate_atr(data)

    st.subheader("Average True Range (ATR)")
    fig_atr = go.Figure()
    fig_atr.add_trace(go.Scatter(x=data['Date'], y=data['ATR'], mode='lines', name='ATR'))
    fig_atr.update_layout(title='Average True Range (ATR)')
    st.plotly_chart(fig_atr, use_container_width=True)

    # Stochastic Oscillator
    def calculate_stochastic_oscillator(data, window=14):
        data['Lowest_Low'] = data['Low'].rolling(window=window).min()
        data['Highest_High'] = data['High'].rolling(window=window).max()
        data['%K'] = 100 * ((data['Close'] - data['Lowest_Low']) / (data['Highest_High'] - data['Lowest_Low']))
        data['%D'] = data['%K'].rolling(window=3).mean()
        return data

    data = calculate_stochastic_oscillator(data)

    st.subheader("Stochastic Oscillator")
    fig_stochastic = go.Figure()
    fig_stochastic.add_trace(go.Scatter(x=data['Date'], y=data['%K'], mode='lines', name='%K'))
    fig_stochastic.add_trace(go.Scatter(x=data['Date'], y=data['%D'], mode='lines', name='%D'))
    fig_stochastic.update_layout(title='Stochastic Oscillator', yaxis=dict(range=[0, 100]))
    st.plotly_chart(fig_stochastic, use_container_width=True)

    # Chaikin Money Flow (CMF)
    def calculate_cmf(data, window=20):
        data['Money_Flow_Multiplier'] = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low'])
        data['Money_Flow_Volume'] = data['Money_Flow_Multiplier'] * data['Volume']
        data['CMF'] = data['Money_Flow_Volume'].rolling(window=window).sum() / data['Volume'].rolling(window=window).sum()
        return data

    data = calculate_cmf(data)

    st.subheader("Chaikin Money Flow (CMF)")
    fig_cmf = go.Figure()
    fig_cmf.add_trace(go.Scatter(x=data['Date'], y=data['CMF'], mode='lines', name='CMF'))
    fig_cmf.update_layout(title='Chaikin Money Flow (CMF)')
    st.plotly_chart(fig_cmf, use_container_width=True)
    
    # Binning
    bin_feature = st.sidebar.selectbox("Select feature for binning:", data.columns)
    num_bins = st.sidebar.slider("Number of bins:", min_value=2, max_value=20, value=10)
    binned_data, bin_edges = np.histogram(data[bin_feature].astype(float), bins=num_bins)

    # Normalization
    normalize_data = st.sidebar.checkbox("Normalize data")
    if normalize_data:
        numerical_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        if not numerical_columns:
            st.error("No numerical columns found for normalization.")
            data_normalized = data
        else:
            scaler = MinMaxScaler()
            data_normalized = pd.DataFrame(scaler.fit_transform(data[numerical_columns]), columns=numerical_columns)
    else:
        data_normalized = data

    # Sampling
    sampling_technique = st.sidebar.selectbox("Sampling technique:", ["None", "Random Sampling", "Stratified Sampling"])
    if sampling_technique == "Random Sampling":
        sample_size = st.sidebar.slider("Sample size:", min_value=1, max_value=len(data), value=len(data)//2)
        sampled_data = data.sample(sample_size)
    elif sampling_technique == "Stratified Sampling":
        stratify_by = st.sidebar.selectbox("Select feature for stratification:", data.columns)
        if stratify_by not in data.columns:
            st.error(f"Selected column '{stratify_by}' does not exist in the dataset.")
            sampled_data = data
        else:
            if sample_size > len(data):
                st.error("Sample size cannot be larger than the dataset size.")
                sample_size = len(data)
            sampled_data = data.groupby(stratify_by, group_keys=False).apply(lambda x: x.sample(min(len(x), sample_size)))
    else:
        sampled_data = data

    # Display results
    st.subheader("Binned Data")
    st.bar_chart(pd.DataFrame({"Bin Edges": bin_edges[:-1], "Frequency": binned_data}))

    st.subheader("Normalized Data")
    st.write(data_normalized)

    st.subheader("Sampled Data")
    st.write(sampled_data)

    # Sidebar options for selecting features
    selected_features = st.sidebar.multiselect("Select features:", ['VWAP', 'Open', 'High', 'Low', 'Adj Close'])

    # Split data into features and target
    X = data[selected_features]
    y = data['Close']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Prediction
    def predict_close_price(features):
        return model.predict([features])[0]

    # Get user input for feature values
    user_input_features = {}
    for feature in selected_features:
        user_input_features[feature] = st.sidebar.number_input(f"Enter value for {feature}:", value=0.0)

    # Predict close price   
    predicted_close = predict_close_price([user_input_features[feature] for feature in selected_features])

    # Display predicted close price
    st.subheader("Predicted Close Price")
    st.write(f"The predicted close price is: {predicted_close}")

    # Model evaluation
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Display model accuracy
    st.subheader("Model Accuracy")
    st.write(f"Mean Squared Error: {mse}")
    st.write(f"R-squared Score: {r2}")

    # Time Series Plots with Animation
    st.subheader("Closing Price Over Time (with Animation)")
    fig_close_animated = go.Figure()
    fig_close_animated.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close'))
    fig_close_animated.update_layout(title='Closing Price Over Time (with Animation)')
    fig_close_animated.update_traces(
        line=dict(color='blue', width=1),
        selector=dict(mode='lines')
    )
    fig_close_animated.update_layout(
        xaxis=dict(range=[data['Date'].min(), data['Date'].max()]),
        yaxis=dict(range=[data['Close'].min(), data['Close'].max()]),
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }
        ]
    )

    frames = [go.Frame(data=[go.Scatter(x=data['Date'][:k+1], y=data['Close'][:k+1])], name=str(k)) for k in range(1, len(data['Date']))]
    fig_close_animated.frames = frames

    st.plotly_chart(fig_close_animated, use_container_width=True)

    st.subheader("Volume Traded Over Time (with Animation)")
    fig_volume_animated = go.Figure()
    fig_volume_animated.add_trace(go.Bar(x=data['Date'], y=data['Volume'], name='Volume'))
    fig_volume_animated.update_layout(title='Volume Traded Over Time (with Animation)')
    fig_volume_animated.update_traces(
        selector=dict(type='bar')
    )
    fig_volume_animated.update_layout(
        xaxis=dict(range=[data['Date'].min(), data['Date'].max()]),
        yaxis=dict(range=[0, data['Volume'].max()]),
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }
        ]
    )

    frames = [go.Frame(data=[go.Bar(x=data['Date'][:k+1], y=data['Volume'][:k+1])], name=str(k)) for k in range(1, len(data['Date']))]
    fig_volume_animated.frames = frames

    st.plotly_chart(fig_volume_animated, use_container_width=True)

    # 3-D Interactive plot for high, low and close prices
    st.subheader("3D Scatter Plot of High, Low, and Close Prices")
    fig_3d_scatter = go.Figure(
        data=[go.Scatter3d(
            x=data['High'],
            y=data['Low'],
            z=data['Close'],
            mode='markers',
            marker=dict(
                size=5,
                color=data['Close'],                # set color to an array/list of desired values
                colorscale='Viridis',                # choose a colorscale
                opacity=0.8
            )
        )]
    )

    fig_3d_scatter.update_layout(
        title="3D Scatter Plot of High, Low, and Close Prices",
        scene=dict(
            xaxis_title='High Price',
            yaxis_title='Low Price',
            zaxis_title='Close Price'
        )
    )
    st.plotly_chart(fig_3d_scatter, use_container_width=True)

    st.subheader("3D Surface Plot of Close Prices Over Time")

    # 3-D Surface plot of close prices over time
    # Create mesh grid for dates and indices
    dates = pd.to_datetime(data['Date']).astype(int) / 10**9  # Convert dates to numeric timestamp
    X, Y = np.meshgrid(dates, data.index)
    Z = data['Close'].values

    fig_3d_surface = go.Figure(
        data=[go.Surface(z=Z, x=X, y=Y)]
    )

    fig_3d_surface.update_layout(
        title="3D Surface Plot of Close Prices Over Time",
        scene=dict(
            xaxis_title='Date',
            yaxis_title='Index',
            zaxis_title='Close Price'
        )
    )
    st.plotly_chart(fig_3d_surface, use_container_width=True)


if __name__ == '__main__':
    main()