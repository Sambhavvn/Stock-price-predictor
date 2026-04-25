import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

from data_loader import fetch_stock_data
from features import add_technical_indicators, create_target
from backtest import backtest_strategy


def prepare_tabular_data(df, feature_cols, target_col):
    df = df.copy()
    df = df.dropna()

    X = df[feature_cols].values
    y = df[target_col].values

    return X, y


def load_multi_stock_data(tickers, feature_cols):
    data_by_stock = {}

    for ticker in tickers:
        df = fetch_stock_data(ticker)
        df = add_technical_indicators(df)
        df = create_target(df)

        df = df.dropna()

        X, y = prepare_tabular_data(df, feature_cols, 'Target')

        data_by_stock[ticker] = (X, y)

    return data_by_stock


def train():
    print("Using XGBoost model (Production Version)")

    feature_cols = [
        'Close', 'MA10', 'MA50', 'EMA10',
        'Returns', 'Volatility', 'RSI',
        'MACD', 'MACD_Signal', 'MACD_Hist',
        'Volume_Change', 'Volume_Ratio',
        'Momentum'
    ]

    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]

    data_by_stock = load_multi_stock_data(tickers, feature_cols)

    # ---------------------------
    # TIME-BASED SPLIT
    # ---------------------------
    X_train_all, y_train_all = [], []
    X_test_all, y_test_all = [], []

    for ticker in tickers:
        X, y = data_by_stock[ticker]

        split = int(0.8 * len(X))

        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        X_train_all.append(X_train)
        y_train_all.append(y_train)

        X_test_all.append(X_test)
        y_test_all.append(y_test)

    X_train = np.vstack(X_train_all)
    y_train = np.concatenate(y_train_all)

    X_test = np.vstack(X_test_all)
    y_test = np.concatenate(y_test_all)

    # ---------------------------
    # MODEL
    # ---------------------------
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)

    # ---------------------------
    # SAVE MODEL (IMPORTANT)
    # ---------------------------
    import os
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/xgb_model.pkl")
    print("\n✅ Model saved to models/xgb_model.pkl")

    # ---------------------------
    # FEATURE IMPORTANCE
    # ---------------------------
    print("\n===== FEATURE IMPORTANCE =====")
    importances = model.feature_importances_

    feature_importance = list(zip(feature_cols, importances))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    for feature, score in feature_importance:
        print(f"{feature}: {score:.4f}")

    # ---------------------------
    # EVALUATION
    # ---------------------------
    probs = model.predict_proba(X_test)[:, 1]
    threshold = 0.6
    y_pred = (probs > threshold).astype(int)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nReport:\n", classification_report(y_test, y_pred, zero_division=0))

    # ---------------------------
    # MULTI-STOCK BACKTEST
    # ---------------------------
    print("\n===== MULTI-STOCK BACKTEST =====")

    portfolio_equity = None

    for ticker in tickers:
        print(f"\n--- Testing on {ticker} ---")

        X, y = data_by_stock[ticker]

        split = int(0.8 * len(X))
        X_test_stock = X[split:]

        probs_bt = model.predict_proba(X_test_stock)[:, 1]

        df_test = fetch_stock_data(ticker)
        df_test = add_technical_indicators(df_test)
        df_test = create_target(df_test)
        df_test = df_test.dropna()

        df_test = df_test.iloc[-len(probs_bt):]

        final_balance, sharpe, drawdown, equity = backtest_strategy(df_test, probs_bt)

        print(f"Final Balance: {final_balance:.2f}")
        print(f"Profit: {final_balance - 10000:.2f}")
        print(f"Sharpe Ratio: {sharpe:.4f}")
        print(f"Max Drawdown: {drawdown:.4f}")

        equity = np.array(equity)

        if portfolio_equity is None:
            portfolio_equity = equity
        else:
            min_len = min(len(portfolio_equity), len(equity))
            portfolio_equity = portfolio_equity[:min_len] + equity[:min_len]

    # ---------------------------
    # PORTFOLIO PERFORMANCE
    # ---------------------------
    print("\n===== PORTFOLIO PERFORMANCE =====")

    portfolio_equity = portfolio_equity / len(tickers)

    returns = np.diff(portfolio_equity) / portfolio_equity[:-1]

    if len(returns) > 1 and np.std(returns) != 0:
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
    else:
        sharpe = 0

    peak = np.maximum.accumulate(portfolio_equity)
    drawdown = (portfolio_equity - peak) / peak
    max_drawdown = np.min(drawdown)

    final_balance = portfolio_equity[-1]

    print(f"Final Portfolio Balance: {final_balance:.2f}")
    print(f"Total Profit: {final_balance - 10000:.2f}")
    print(f"Portfolio Sharpe Ratio: {sharpe:.4f}")
    print(f"Portfolio Max Drawdown: {max_drawdown:.4f}")


if __name__ == "__main__":
    train()