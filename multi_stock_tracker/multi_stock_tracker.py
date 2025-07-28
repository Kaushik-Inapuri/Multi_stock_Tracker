import streamlit as st
from streamlit_autorefresh import st_autorefresh
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import pytz

# ========== Refresh every 30 seconds ==========
st_autorefresh(interval=30000, limit=None, key="datarefresh")

# ========== Page settings ==========
st.set_page_config(page_title="📈 Multi-Stock Tracker", layout="wide")

# ========== Sticky Header Styling ==========
# Sticky header styling
st.markdown("""
    <style>
        .market-status {
            position: fixed;
            top: 0;
            width: 100%;
            text-align: center;
            background-color: #262730;
            color: white;
            padding: 10px 0;
            font-size: 18px;
            z-index: 9999;
            border-bottom: 2px solid #444;
        }
        .block-container {
            padding-top: 70px !important;  /* prevent content hiding under banner */
        }
    </style>
""", unsafe_allow_html=True)
    # Auto-refresh only the header part every second
st_autorefresh(interval=1000, key="header_refresh")


# ========== Market Status Logic ==========
def get_market_status():
    now = datetime.now(pytz.timezone('Asia/Kolkata'))
    open_time = now.replace(hour=9, minute=15, second=0, microsecond=0)
    close_time = now.replace(hour=15, minute=30, second=0, microsecond=0)

    if now.weekday() >= 5:
        return "❌ Market is Closed (Weekend)", None
    elif now < open_time:
        countdown = open_time - now
        return "⏳ Market Opens In:", countdown
    elif now >= close_time:
        return "❌ Market is Closed", None
    else:
        countdown = close_time - now
        return "✅ Market is Open — Closes In:", countdown

# Get market status and countdown
status_text, countdown = get_market_status()

# Show sticky live header with countdown
if countdown:
    st.markdown(f"""
        <div class='market-status'>
            {status_text} ⏱️ {str(countdown).split('.')[0]}
        </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
        <div class='market-status'>
            {status_text}
        </div>
    """, unsafe_allow_html=True)


# ========== Title ==========
st.title("📊 Live Multi-Stock Portfolio Tracker")

# ========== Input for stock tickers ==========
symbols = st.text_input("Enter stock tickers (comma-separated):", value="TCS.NS, SBIN.NS, IDEA.NS")
symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]

# ========== Input for investment details ==========
st.subheader("💼 Investment Details")
portfolio = []
for symbol in symbol_list:
    with st.expander(f"{symbol} – Investment Details"):
        buy_price = st.number_input(f"Buy Price for {symbol} (₹)", min_value=0.0, step=0.1, key=f"buy_{symbol}")
        quantity = st.number_input(f"Quantity for {symbol}", min_value=0, step=1, key=f"qty_{symbol}")
        portfolio.append({"symbol": symbol, "buy_price": buy_price, "quantity": quantity})

# ========== RSI Function ==========
@st.cache_data(ttl=30)
def compute_rsi(data, window=14):
    delta = data.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

# ========== Process & Display ==========
st.subheader("📈 Stock Details")
total_investment = 0
total_current_value = 0

for stock in portfolio:
    symbol = stock["symbol"]
    st.markdown(f"---\n### {symbol}")

    try:
        stock_data = yf.Ticker(symbol)
        df = stock_data.history(period="2d", interval="5m")

        if df.empty or len(df) < 2:
            st.warning(f"No intraday data available for {symbol}")
            continue

        df = df.dropna().reset_index()
        time_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
        df = df[[time_col, 'Open', 'High', 'Low', 'Close', 'Volume']]
        df['TimeIndex'] = np.arange(len(df))
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['EMA_5'] = df['Close'].ewm(span=5).mean()
        df['RSI_14'] = compute_rsi(df['Close'])
        df.dropna(inplace=True)

        # ===== Prediction =====
        features = ['TimeIndex', 'SMA_5', 'EMA_5', 'RSI_14', 'Volume']
        X = df[features]
        y = df['Close']
        model = LinearRegression()
        model.fit(X, y)

        future = df[features].iloc[[-1]].copy()
        future['TimeIndex'] += 1
        predicted_next = model.predict(future)[0]

        closing_input = future.copy()
        closing_input['TimeIndex'] = 75
        predicted_close = model.predict(closing_input)[0]

        # ===== Price Info =====
        current_price = df.iloc[-1]['Close']
        opening_price = df.iloc[0]['Open']
        intraday_momentum = current_price - opening_price
        momentum_label = "📈 Up" if intraday_momentum > 0 else "🔻 Down"
        momentum_text = f"{momentum_label} ₹{abs(intraday_momentum):.2f} from open (₹{opening_price:.2f})"

        prev_day_close = df.iloc[0]['Close']

        st.metric("📍 Current Price", f"₹{current_price:.2f}", delta=momentum_text)
        st.metric("🔮 Next Estimate (5min)", f"₹{predicted_next:.2f}")
        st.metric("🕔 Estimated Closing", f"₹{predicted_close:.2f}")
        st.metric("📉 Prev Day Close", f"₹{prev_day_close:.2f}")

        # ===== Portfolio Profit =====
        invested = stock["buy_price"] * stock["quantity"]
        current_value = current_price * stock["quantity"]
        profit = current_value - invested
        profit_pct = (profit / invested) * 100 if invested != 0 else 0
        total_investment += invested
        total_current_value += current_value

        st.metric("💸 Invested", f"₹{invested:.2f}")
        st.metric("💼 Current Value", f"₹{current_value:.2f}", delta=f"₹{profit:.2f}")

        if profit > 0:
            st.success("✅ Consider Selling")
        elif profit < -1000 or profit_pct < -5:
            st.error("❌ High Loss – Consider Exiting")
        else:
            st.warning("🔁 Hold / Watch Closely")

        # ===== Candlestick Chart =====
        fig = go.Figure(data=[go.Candlestick(
            x=df[time_col],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            increasing_line_color='green',
            decreasing_line_color='red'
        )])
        fig.update_layout(
            title=f"{symbol} – Intraday Candlestick",
            xaxis_title='Time',
            yaxis_title='Price (₹)',
            xaxis_rangeslider_visible=False,
            template='plotly_dark',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error loading {symbol}: {e}")

# ========== Portfolio Summary ==========
st.markdown("---")
st.header("📦 Portfolio Summary")
total_profit = total_current_value - total_investment
st.metric("💰 Total Invested", f"₹{total_investment:.2f}")
st.metric("📊 Current Value", f"₹{total_current_value:.2f}", delta=f"₹{total_profit:.2f}")
if total_profit > 0:
    st.success("✅ You're in profit!")
elif total_profit < -1000:
    st.error("❌ High Portfolio Loss – Reconsider Strategy")
else:
    st.warning("🔁 Portfolio in loss.")

# ========== Explore Another Stock ==========
st.markdown("---")
st.header("🔍 Explore Another Stock (Optional)")
explore_symbol = st.text_input("Enter a stock ticker (e.g., INFY.NS, RELIANCE.NS):", value="")

if explore_symbol:
    try:
        exp_data = yf.Ticker(explore_symbol)
        exp_df = exp_data.history(period="1d", interval="5m")
        exp_df = exp_df.dropna().reset_index()

        st.subheader(f"📊 {explore_symbol.upper()} – Intraday Snapshot")
        current_exp_price = exp_df['Close'].iloc[-1]
        st.metric("💹 Current Price", f"₹{current_exp_price:.2f}")

        fig = go.Figure(data=[go.Candlestick(
            x=exp_df['Datetime'],
            open=exp_df['Open'],
            high=exp_df['High'],
            low=exp_df['Low'],
            close=exp_df['Close'],
            increasing_line_color='green',
            decreasing_line_color='red'
        )])
        fig.update_layout(
            title=f"{explore_symbol.upper()} – Intraday Candlestick",
            xaxis_title='Time',
            yaxis_title='Price (₹)',
            xaxis_rangeslider_visible=False,
            template='plotly_dark',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error loading data for {explore_symbol.upper()}: {e}")
