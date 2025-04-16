# XAUUSD Real-Time Trading Analyzer

A professional Python-based tool for real-time technical analysis of XAUUSD (Gold/USD) for day trading purposes. This script downloads market data, applies technical indicators, generates trading signals, and provides risk management calculations to assist with trading decisions.

## Features

- Real-time market data retrieval from Yahoo Finance
- Multiple technical indicators calculation (RSI, MACD, EMAs, Bollinger Bands)
- Support and resistance level identification
- Automated trading signals generation (LONG/SHORT)
- Stop Loss and Take Profit level calculation
- Professional risk management implementation
- Comprehensive visualization of price charts with indicators
- Continuous market monitoring at customizable intervals

## Requirements

- Python 3.7+
- Dependencies:
  - pandas
  - numpy
  - matplotlib
  - yfinance
  - ta (Technical Analysis Library)

## Installation

1. Clone this repository or download the script
2. Install required packages:

```bash
pip install pandas numpy matplotlib yfinance ta
```

3. Run the script:

```bash
python xauusd_analyzer.py
```

## How It Works

### Data Collection
The script retrieves XAUUSD historical data from Yahoo Finance using the "GC=F" ticker symbol, with 1-hour timeframes suitable for day trading strategies.

### Technical Indicators
The following indicators are calculated and used in the analysis:
- **Relative Strength Index (RSI)**: Measures momentum and identifies overbought/oversold conditions
- **Moving Average Convergence Divergence (MACD)**: Trend-following momentum indicator
- **Exponential Moving Averages (EMAs)**: Short (9), Medium (21), and Long (50) periods
- **Bollinger Bands**: Volatility-based bands around a simple moving average
- **Average True Range (ATR)**: Measures market volatility
- **Support and Resistance Levels**: Dynamic identification based on recent price action

### Trading Strategy
The implemented strategy generates entry signals based on:

1. **LONG Signal Conditions**:
   - MACD crosses above its signal line
   - RSI below 70 (not overbought)
   - Short EMA above Medium EMA (uptrend confirmation)

2. **SHORT Signal Conditions**:
   - MACD crosses below its signal line
   - RSI above 30 (not oversold)
   - Short EMA below Medium EMA (downtrend confirmation)

### Risk Management
The script incorporates professional risk management principles:
- **Stop Loss Calculation**: Based on 1.5 × ATR or recent significant price levels
- **Take Profit Calculation**: Based on a 2:1 reward-to-risk ratio
- **Position Sizing**: Automatically calculated based on account balance and risk percentage
- **Risk Per Trade**: Default 1% of account capital

## Usage

### Basic Usage
Run the script to perform a single analysis of current market conditions:

```python
analyzer = XAUUSDAnalyzer()
analyzer.perform_analysis()
```

### Real-Time Monitoring
For continuous monitoring and updated signals at regular intervals:

```python
analyzer = XAUUSDAnalyzer()
analyzer.start_real_time_monitoring(interval_minutes=15)
```

The default interval is 15 minutes, but you can customize this value.

### Output
The script provides:
1. **Console Output**:
   - Current price and trend analysis
   - Technical indicator values
   - Trading signal (if available)
   - Entry, Stop Loss, and Take Profit levels
   - Position sizing recommendations for different account balances
   - Key support and resistance zones

2. **Visualization**:
   - Price chart with EMAs and Bollinger Bands
   - MACD indicator chart
   - RSI indicator chart
   - Trading signals marked on the chart
   - Stop Loss and Take Profit levels visualized as horizontal lines

## Customization

You can modify the following parameters in the `XAUUSDAnalyzer` class:
- `timeframe`: Data interval (default: "1h")
- `rsi_period`: RSI calculation period (default: 14)
- `macd_fast`, `macd_slow`, `macd_signal`: MACD parameters (default: 12, 26, 9)
- `ema_short`, `ema_medium`, `ema_long`: EMA periods (default: 9, 21, 50)
- `atr_period`: ATR calculation period (default: 14)
- `risk_reward_ratio`: Take Profit to Stop Loss ratio (default: 2)
- `risk_percentage`: Risk per trade as percentage (default: 1%)

## Important Notes

- This script is provided for educational and informational purposes only.
- Trading financial instruments involves significant risk of loss.
- Past performance is not indicative of future results.
- Always conduct your own analysis and consider consulting with a licensed financial advisor before making trading decisions.
- The script does not guarantee profitable trades or accurate signals.

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.