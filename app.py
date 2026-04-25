import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

from src.data_loader import fetch_stock_data
from src.features import add_technical_indicators

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "models/xgb_model.pkl"

TRANSACTION_COST = 0.001
SLIPPAGE = 0.0005
RISK_PER_TRADE = 0.02
STOP_LOSS = -0.02
TAKE_PROFIT = 0.04

feature_cols = [
    'Close', 'MA10', 'MA50', 'EMA10',
    'Returns', 'Volatility', 'RSI',
    'MACD', 'MACD_Signal', 'MACD_Hist',
    'Volume_Change', 'Volume_Ratio',
    'Momentum'
]

model = joblib.load(MODEL_PATH)

st.set_page_config(page_title="AI Trading Platform", layout="wide")

# -----------------------------
# SESSION STATE
# -----------------------------
if "alerts" not in st.session_state:
    st.session_state.alerts = []

# -----------------------------
# STOCK LIST
# -----------------------------
@st.cache_data
def get_stocks():
    return [
        "AAPL","MSFT","GOOG","AMZN","TSLA","META","NVDA","NFLX","AMD","INTC",
        "BABA","ORCL","IBM","ADBE","CRM","PYPL","UBER","LYFT","SHOP","QCOM",
        "CSCO","INTU","TXN","AVGO","NOW","SPOT","SQ","SNOW","COIN","ZM"
    ]

# -----------------------------
# SIGNAL
# -----------------------------
def get_signal(prob):
    if prob > 0.75:
        return "BUY", "STRONG"
    elif prob > 0.6:
        return "BUY", "WEAK"
    elif prob < 0.25:
        return "SELL", "STRONG"
    elif prob < 0.4:
        return "SELL", "WEAK"
    else:
        return "HOLD", "NEUTRAL"

# -----------------------------
# SIMULATION (TIME BASED)
# -----------------------------
def simulate(df, initial_capital):

    balance = initial_capital
    equity = []
    returns_list = []
    timeline = []

    for i in range(len(df) - 2):

        row = df.iloc[i]

        X = np.array([row[feature_cols].values])
        prob = model.predict_proba(X)[0][1]

        signal, _ = get_signal(prob)
        confidence = abs(prob - 0.5)

        if confidence < 0.1 or signal == "HOLD":
            equity.append(balance)
            timeline.append(df.index[i])
            continue

        entry = df.iloc[i + 1]['Open']
        exit_ = df.iloc[i + 2]['Open']

        change = (exit_ - entry) / entry

        if signal == "SELL":
            change = -change

        change -= (TRANSACTION_COST + SLIPPAGE)

        if change < STOP_LOSS:
            change = STOP_LOSS
        elif change > TAKE_PROFIT:
            change = TAKE_PROFIT

        pnl = balance * RISK_PER_TRADE * change
        balance += pnl

        returns_list.append(change)
        equity.append(balance)
        timeline.append(df.index[i])

    return equity, returns_list, timeline

# -----------------------------
# NAVIGATION
# -----------------------------
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to", ["📈 Analysis", "⚡ Live Scanner"])

# =============================
# ANALYSIS PAGE
# =============================
if page == "📈 Analysis":

    st.title("📈 Stock Analysis")

    ticker = st.selectbox("Select Stock", get_stocks())

    capital = st.number_input("💰 Investment Amount", value=10000, step=1000)

    try:
        df = fetch_stock_data(ticker)
        df = add_technical_indicators(df)
        df = df.ffill().bfill().dropna().reset_index(drop=True)

        latest = df.iloc[-1]

        X = np.array([latest[feature_cols].values])
        prob = model.predict_proba(X)[0][1]

        signal, strength = get_signal(prob)
        confidence = abs(prob - 0.5)

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Probability", f"{prob:.3f}")
        col2.metric("Confidence", f"{confidence:.2f}")
        col3.metric("Price", f"{latest['Close']:.2f}")
        col4.metric("Signal", f"{signal} ({strength})")

        # Chart
        st.subheader("📉 Price Chart")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close'))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='MA50'))

        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        # -----------------------------
        # STRATEGY PERFORMANCE
        # -----------------------------
        st.subheader("📊 Strategy Performance")

        equity, returns, timeline = simulate(df, capital)

        if equity:

            final_balance = equity[-1]
            profit = final_balance - capital

            returns_series = pd.Series(returns)

            win_rate = (returns_series > 0).mean()
            sharpe = (returns_series.mean() / returns_series.std()) * np.sqrt(252) if returns_series.std() != 0 else 0

            peak = pd.Series(equity).cummax()
            drawdown = (pd.Series(equity) - peak) / peak
            max_dd = drawdown.min()

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Final Balance", f"{final_balance:.2f}")
            col2.metric("Profit", f"{profit:.2f}")
            col3.metric("Win Rate", f"{win_rate:.2%}")
            col4.metric("Max Drawdown", f"{max_dd:.2%}")

            st.metric("Sharpe Ratio", f"{sharpe:.2f}")

            # Equity Curve
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=timeline, y=equity, name="Equity Curve"))
            fig2.update_layout(template="plotly_dark")
            st.plotly_chart(fig2, use_container_width=True)

            # -----------------------------
            # TIME-BASED PERFORMANCE
            # -----------------------------
            st.subheader("⏱ Time-Based Performance")

            equity_series = pd.Series(equity)

            def get_value(days):
                if len(equity_series) > days:
                    return equity_series.iloc[days]
                return equity_series.iloc[-1]

            col1, col2, col3 = st.columns(3)

            val_30 = get_value(30)
            val_90 = get_value(90)
            val_180 = get_value(180)

            col1.metric("30 Days", f"{val_30:.2f}", f"{val_30 - capital:.2f}")
            col2.metric("90 Days", f"{val_90:.2f}", f"{val_90 - capital:.2f}")
            col3.metric("180 Days", f"{val_180:.2f}", f"{val_180 - capital:.2f}")

    except:
        st.error("Error loading stock")

# =============================
# LIVE SCANNER
# =============================
elif page == "⚡ Live Scanner":

    st.title("⚡ Live Scanner")

    stocks = get_stocks()
    num = st.slider("Stocks to Monitor", 10, len(stocks), 20)

    if st.button("🔄 Refresh"):

        results = []

        for t in stocks[:num]:
            try:
                df = fetch_stock_data(t)
                df = add_technical_indicators(df)
                df = df.ffill().bfill().dropna()

                X = np.array([df.iloc[-1][feature_cols].values])
                prob = model.predict_proba(X)[0][1]

                signal, strength = get_signal(prob)
                confidence = abs(prob - 0.5)

                if confidence < 0.1:
                    continue

                results.append({
                    "Ticker": t,
                    "Signal": f"{signal} ({strength})",
                    "Probability": round(prob, 3),
                    "Confidence": round(confidence, 3)
                })

                if signal != "HOLD":
                    st.session_state.alerts.append({
                        "Ticker": t,
                        "Signal": signal,
                        "Confidence": round(confidence, 3),
                        "Time": pd.Timestamp.now()
                    })

            except:
                continue

        if results:
            df_results = pd.DataFrame(results)
            df_results = df_results.sort_values(by="Confidence", ascending=False)

            st.subheader("📡 Opportunities")
            st.dataframe(df_results, use_container_width=True)

    st.subheader("📜 Alert History")

    if st.session_state.alerts:
        alerts_df = pd.DataFrame(st.session_state.alerts)
        alerts_df = alerts_df.sort_values(by="Time", ascending=False)

        st.dataframe(alerts_df.head(20), use_container_width=True)
    else:
        st.info("No alerts yet")

# -----------------------------
# FOOTER
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.success("System Running")