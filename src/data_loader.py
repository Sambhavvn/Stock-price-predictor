import yfinance as yf
import pandas as pd


def fetch_stock_data(ticker):
    df = yf.download(
        ticker,
        period="5y",
        interval="1d",
        progress=False
    )

    # Reset index
    df = df.reset_index()

    # 🔥 FIX: Flatten MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]

    # 🔥 FIX: Normalize column names
    rename_map = {}
    for col in df.columns:
        col_lower = col.lower()

        if 'close' in col_lower:
            rename_map[col] = 'Close'
        elif 'volume' in col_lower:
            rename_map[col] = 'Volume'
        elif 'open' in col_lower:
            rename_map[col] = 'Open'
        elif 'high' in col_lower:
            rename_map[col] = 'High'
        elif 'low' in col_lower:
            rename_map[col] = 'Low'
        elif 'date' in col_lower:
            rename_map[col] = 'Date'

    df = df.rename(columns=rename_map)

    # 🔥 FINAL SAFETY: ensure single columns (not DataFrame)
    for col in ['Close', 'Volume', 'Open', 'High', 'Low']:
        if col in df.columns and isinstance(df[col], pd.DataFrame):
            df[col] = df[col].iloc[:, 0]

    return df