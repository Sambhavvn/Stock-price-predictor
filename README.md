# AI Stock Trading Platform

A machine learning-powered stock analysis and trading simulation platform that uses XGBoost to predict buy/sell signals based on technical indicators. Features an interactive Streamlit dashboard for real-time analysis and backtesting.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35.0-red.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-green.svg)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/Sambhavvn/Stock-price-predictor/blob/main/LISCENSE)

## Features

- **AI-Powered Predictions**: XGBoost classification model trained on technical indicators
- **Technical Analysis**: RSI, MACD, Moving Averages (MA10, MA50, EMA10), Volume indicators, Momentum
- **Interactive Dashboard**: Real-time stock analysis with Plotly visualizations
- **Strategy Simulation**: Backtesting with realistic trading parameters:
  - Transaction costs (0.1%)
  - Slippage (0.05%)
  - Risk per trade (2%)
  - Stop-loss (-2%) and Take-profit (+4%)
- **Live Scanner**: Monitor 30+ stocks for trading opportunities
- **Performance Metrics**: Win rate, Sharpe ratio, max drawdown, equity curves

## Tech Stack

- **Python 3.8+**
- **Streamlit** - Web application framework
- **XGBoost** - Machine learning classifier
- **scikit-learn** - Model evaluation and preprocessing
- **yFinance** - Stock market data retrieval
- **Plotly** - Interactive charts
- **Pandas/NumPy** - Data manipulation

## Installation

### Prerequisites
- Python 3.8 or higher
- Git (optional, for cloning)

### Setup

1. **Clone or download the repository**
```bash
git clone https://github.com/yourusername/stock-predictor.git
cd stock-predictor
```

2. **Create virtual environment (recommended)**
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Dependencies
```
streamlit==1.35.0
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.4.2
xgboost==2.0.3
joblib==1.4.2
yfinance==0.2.40
plotly==5.22.0
lxml==5.2.1
html5lib==1.1
requests==2.32.3
```

## Usage

### Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Using the Dashboard

1. **Analysis Page**
   - Select a stock from the dropdown (30+ stocks available)
   - Enter your investment amount
   - View AI-generated signals (BUY/SELL/HOLD) with confidence scores
   - Analyze price charts with moving averages
   - Run strategy simulation to see backtested performance

2. **Live Scanner**
   - Select number of stocks to monitor
   - Click "Refresh" to scan for opportunities
   - View signal history and alerts

### Retraining the Model

To retrain the model with new data or different parameters:

```bash
cd src
python train.py
```

This will:
- Download 5 years of data for training stocks (AAPL, MSFT, GOOG, AMZN, TSLA)
- Extract technical indicators
- Train XGBoost classifier with time-based split (80% train, 20% test)
- Save model to `models/xgb_model.pkl`
- Display feature importance and backtest results

## Project Structure

```
stock-predictor/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── models/
│   └── xgb_model.pkl     # Trained XGBoost model
├── data/                 # Data storage (empty by default)
└── src/
    ├── data_loader.py    # yFinance data fetching
    ├── features.py       # Technical indicators
    ├── train.py          # Model training script
    ├── utils.py          # Utility functions
    └── backtest.py       # Backtesting logic
```

## Model Training Details

### Features Used
| Feature | Description |
|---------|-------------|
| Close | Closing price |
| MA10 | 10-day Moving Average |
| MA50 | 50-day Moving Average |
| EMA10 | 10-day Exponential MA |
| Returns | Daily returns |
| Volatility | Rolling volatility |
| RSI | Relative Strength Index |
| MACD | MACD line |
| MACD_Signal | MACD signal line |
| MACD_Hist | MACD histogram |
| Volume_Change | Volume change |
| Volume_Ratio | Volume ratio |
| Momentum | Price momentum |

### Model Configuration
- **Algorithm**: XGBoost Classifier
- **Training Data**: 5 years of historical data
- **Stocks**: AAPL, MSFT, GOOG, AMZN, TSLA
- **Train/Test Split**: 80/20 time-based split
- **Target**: Binary classification (Price up/down next day)

### Hyperparameters
```python
n_estimators=200
max_depth=5
learning_rate=0.05
subsample=0.8
colsample_bytree=0.8
```

### Signal Thresholds
| Probability | Signal | Strength |
|-------------|--------|----------|
| > 0.75 | BUY | STRONG |
| 0.60 - 0.75 | BUY | WEAK |
| 0.40 - 0.60 | HOLD | NEUTRAL |
| 0.25 - 0.40 | SELL | WEAK |
| < 0.25 | SELL | STRONG |

## Risk Disclaimer

**IMPORTANT**: This project is for educational purposes only. The predictions made by this model should not be considered as financial advice. Always do your own research and consult with financial advisors before making investment decisions. Past performance does not guarantee future results.

## Future Enhancements

- [ ] Add more technical indicators (Bollinger Bands, Fibonacci)
- [ ] Implement portfolio optimization
- [ ] Add news sentiment analysis
- [ ] Real-time data streaming
- [ ] Paper trading integration
- [ ] Multi-timeframe analysis
- [ ] Add more ML models (LSTM, Random Forest)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Acknowledgments

- [yFinance](https://github.com/ranaroussi/yfinance) for stock market data
- [Streamlit](https://streamlit.io/) for the amazing web framework
- [XGBoost](https://xgboost.readthedocs.io/) for the powerful ML library

## Contact

Sambhav v namanna 
sambhavnamanna@gmail.com
