import pandas as pd
import numpy as np


def add_technical_indicators(df):
    df = df.copy()

    # Ensure Close and Volume are single columns
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')

    # -----------------------
    # Moving Averages
    # -----------------------
    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    df['EMA10'] = df['Close'].ewm(span=10).mean()

    # -----------------------
    # Returns & Volatility
    # -----------------------
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(10).std()

    # -----------------------
    # RSI
    # -----------------------
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()

    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))

    # -----------------------
    # MACD
    # -----------------------
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()

    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # -----------------------
    # Volume Features (FIXED)
    # -----------------------
    df['Volume_Change'] = df['Volume'].pct_change()

    volume_ma = df['Volume'].rolling(10).mean()

    # 🔥 FIX: ensure it's scalar division, not dataframe
    df['Volume_Ratio'] = df['Volume'] / (volume_ma + 1e-9)

    # -----------------------
    # Momentum
    # -----------------------
    df['Momentum'] = df['Close'] - df['Close'].shift(10)

    return df


def create_target(df, horizon=1, threshold=0.002):
    df = df.copy()

    future_return = df['Close'].shift(-horizon) / df['Close'] - 1

    df['Target'] = 0
    df.loc[future_return > threshold, 'Target'] = 1

    return df