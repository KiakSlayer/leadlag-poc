# Cross-Asset Lead-Lag Arbitrage System (PoC)

A comprehensive Proof-of-Concept implementation of a cross-asset lead-lag arbitrage trading system that identifies and exploits statistical relationships between cryptocurrencies and equity indices.

## 🎯 Project Overview

This system implements a quantitative trading strategy that:
1. Detects lead-lag relationships between crypto assets and equity indices
2. Uses Z-score based mean-reversion signals
3. Backtests strategies with realistic transaction costs
4. Provides comprehensive performance analysis and visualization

### Supported Assets

**Cryptocurrencies** (via Binance API):
- BTCUSDT (Bitcoin)
- ETHUSDT (Ethereum)
- SOLUSDT (Solana)

**Equity Indices** (via Yahoo Finance):
- ^GSPC (S&P 500)
- ^IXIC (NASDAQ Composite)
- ^SET.BK (SET50 - Thailand Stock Exchange)

## 📁 Project Structure

```
leadlag-poc/
├── core/data_fetcher.py              # Multi-source data collection
├── core/correlation_analyzer.py       # Correlation & lead-lag detection
├── core/crossasset_leadlag_model.py  # Z-score trading model
├── core/backtester.py                # Performance simulation
├── core/visualizer.py                # Plotting & reporting
├── main_crossasset_poc.py       # Main orchestration script
├── requirements.txt             # Python dependencies
├── docker-compose.yml           # Kafka infrastructure (for streaming)
└── README_CROSSASSET.md         # This file
```

## 🔧 Installation

### Prerequisites
- Python 3.8+
- pip package manager
- Internet connection (for data APIs)

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd leadlag-poc
```

2. **Create virtual environment** (recommended)
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Quick Start

Run the complete analysis with default parameters:

```bash
python main_crossasset_poc.py
```

This will:
1. Fetch 5 days of 1-minute data for BTC, ETH, SOL and indices
2. Calculate correlations between all crypto-index pairs
3. Detect lead-lag relationships
4. Run backtests on top 3 pairs
5. Generate visualization reports

### Command-Line Options

```bash
python main_crossasset_poc.py [OPTIONS]

Options:
  --crypto SYMBOLS       Crypto symbols (default: BTCUSDT ETHUSDT SOLUSDT)
  --period PERIOD        Data period (default: 5d)
  --interval INTERVAL    Data interval (default: 1m)
  --window SIZE          Rolling window (default: 60)
  --z-entry THRESHOLD    Entry Z-score (default: 2.0)
  --z-exit THRESHOLD     Exit Z-score (default: 0.5)
  --min-corr VALUE       Min correlation (default: 0.3)
  --max-lag PERIODS      Max lag to test (default: 10)
  --capital AMOUNT       Initial capital (default: 100000)
  --cost FRACTION        Transaction cost (default: 0.001)
  --no-save              Don't save plots
```

### Examples

**Test with different parameters:**
```bash
python main_crossasset_poc.py --window 30 --z-entry 1.5 --capital 50000
```

**Only analyze BTC-SP500 pair:**
```bash
python main_crossasset_poc.py --crypto BTCUSDT --period 7d
```

**Faster analysis (larger interval):**
```bash
python main_crossasset_poc.py --interval 5m --period 1mo
```

## 📊 Module Documentation

### 1. `core/data_fetcher.py`

Collects OHLCV data from multiple sources.

**Key Classes:**
- `DataFetcher`: Unified interface for crypto and equity data

**Key Methods:**
```python
fetcher = DataFetcher()

# Fetch crypto data
df = fetcher.fetch_crypto_ohlcv('BTCUSDT', interval='1m', limit=1000)

# Fetch equity data
df = fetcher.fetch_equity_ohlcv('^GSPC', period='5d', interval='1m')

# Fetch all assets
data = fetcher.fetch_all_assets(
    crypto_symbols=['BTCUSDT', 'ETHUSDT'],
    equity_symbols={'SP500': '^GSPC', 'NASDAQ': '^IXIC'}
)

# Align timestamps
aligned = fetcher.align_timestamps(data, method='inner')

# Get close prices
prices = fetcher.get_close_prices(aligned)
```

### 2. `core/correlation_analyzer.py`

Analyzes correlations and lead-lag relationships.

**Key Classes:**
- `CorrelationAnalyzer`: Statistical analysis of asset pairs

**Key Methods:**
```python
analyzer = CorrelationAnalyzer(prices_df)

# Get correlation matrix
corr_matrix = analyzer.calculate_correlation_matrix()

# Find best pairs
pairs = analyzer.find_best_pairs(
    crypto_assets=['BTCUSDT', 'ETHUSDT'],
    index_assets=['SP500', 'NASDAQ'],
    min_correlation=0.3
)

# Detect lead-lag
lag, max_corr = analyzer.detect_lead_lag('BTCUSDT', 'SP500', max_lag=10)

# Analyze all pairs
analysis = analyzer.analyze_all_pairs(pairs, max_lag=10)
```

### 3. `core/crossasset_leadlag_model.py`

Implements Z-score trading strategy.

**Key Classes:**
- `ModelConfig`: Configuration parameters
- `CrossAssetLeadLagModel`: Lead-lag trading model

**Strategy Logic:**
```python
config = ModelConfig(
    window=60,      # Rolling window
    z_entry=2.0,    # Entry threshold
    z_exit=0.5      # Exit threshold
)

model = CrossAssetLeadLagModel(config)

# Run strategy
signals = model.run_strategy(
    prices=prices_df,
    leader='BTCUSDT',
    lagger='SP500',
    lag=0
)
```

**Signal Generation:**
- `Z > 2.0` → SHORT leader, LONG lagger
- `Z < -2.0` → LONG leader, SHORT lagger
- `|Z| < 0.5` → FLAT (close positions)
- Otherwise → HOLD

### 4. `core/backtester.py`

Simulates trading and calculates performance.

**Key Classes:**
- `BacktestConfig`: Backtest parameters
- `Backtester`: Execution engine
- `PerformanceMetrics`: Metric calculations

**Key Metrics:**
- Total Return
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Win Rate
- Number of Trades

**Usage:**
```python
config = BacktestConfig(
    initial_capital=100000,
    transaction_cost=0.001,  # 0.1%
    position_size=1.0
)

backtester = Backtester(config)

results = backtester.run_backtest(
    signals_df=signals,
    prices=prices,
    leader='BTCUSDT',
    lagger='SP500'
)

# Access results
equity_curve = results['equity_curve']
trades = results['trades']
metrics = results['metrics']
```

### 5. `core/visualizer.py`

Creates comprehensive visualizations.

**Key Classes:**
- `StrategyVisualizer`: Plotting tools

**Generates:**
- Price series comparison
- Z-score evolution with thresholds
- Equity curve with drawdown
- Signal timeline
- Returns distribution

**Usage:**
```python
visualizer = StrategyVisualizer()

fig = visualizer.create_comprehensive_report(
    prices=prices,
    signals_df=signals,
    equity_df=equity_curve,
    metrics=metrics,
    leader='BTCUSDT',
    lagger='SP500',
    save_path='report.png'
)
```

## 📈 Strategy Explanation

### Lead-Lag Relationship

**Concept:** Some assets tend to move before others. If Asset A (leader) moves before Asset B (lagger), we can profit by:
1. Detecting when the spread deviates from normal
2. Betting on mean reversion

### Mathematical Framework

**1. Hedge Ratio (Beta):**
```
β = cov(r_leader, r_lagger) / var(r_lagger)
```

**2. Spread:**
```
spread = r_leader - β × r_lagger
```

**3. Z-Score:**
```
Z = (spread - mean(spread)) / std(spread)
```

**4. Signals:**
- Z > +2.0: Spread too high → SHORT leader, LONG lagger
- Z < -2.0: Spread too low → LONG leader, SHORT lagger
- |Z| < 0.5: Spread normalized → FLAT

### Why This Works

1. **Mean Reversion:** Statistical relationships tend to revert to their mean
2. **Market Efficiency:** Cross-asset relationships are harder to arbitrage than single assets
3. **Diversification:** Trading relationships reduces directional risk

## 📊 Output Examples

### Console Output

```
═══════════════════════════════════════════════════════════════
       Cross-Asset Lead-Lag Arbitrage System (PoC)
═══════════════════════════════════════════════════════════════

📊 STEP 1: DATA COLLECTION
✓ Fetched 1000 1m bars for BTCUSDT
✓ Fetched 1000 1m bars for SP500
✓ Aligned data to 950 common timestamps

🔗 STEP 2: CORRELATION ANALYSIS
Top Correlated Pairs:
  BTCUSDT         <-> SP500     : +0.6234

Lead-Lag Analysis:
  BTCUSDT leads SP500 by 3 periods (correlation: 0.6891)

📈 STEP 3: STRATEGY EXECUTION
  ✓ Backtest Results:
    Total Return:                      +12.45%
    Sharpe Ratio:                        1.85
    Max Drawdown:                       -5.23%
    Number of Trades:                      42
    Win Rate:                          57.14%
```

### Visualization Report

Generated PNG includes:
- **Panel 1:** Normalized price series
- **Panel 2:** Z-score with entry/exit thresholds
- **Panel 3:** Equity curve with drawdown overlay
- **Panel 4:** Trading signals on price chart
- **Panel 5:** Returns distribution histogram
- **Text Box:** Performance metrics summary

## 🧪 Testing Individual Modules

Each module includes a test section that runs when executed directly:

```bash
# Test data fetcher
python core/data_fetcher.py

# Test correlation analyzer
python core/correlation_analyzer.py

# Test model
python core/crossasset_leadlag_model.py

# Test backtester
python core/backtester.py

# Test visualizer
python core/visualizer.py
```

## 🔄 Future Enhancements

### Phase 2 (Streaming Integration)
- [ ] Integrate with existing Kafka pipeline
- [ ] Real-time signal generation
- [ ] Live monitoring dashboard

### Phase 3 (Advanced Features)
- [ ] Machine learning for dynamic parameters
- [ ] Multi-pair portfolio optimization
- [ ] Risk management layer
- [ ] Execution simulation with slippage

### Phase 4 (Production)
- [ ] Paper trading integration
- [ ] Live trading with broker APIs
- [ ] Monitoring and alerting
- [ ] Performance tracking database

## ⚠️ Important Notes

### Data Limitations

1. **Yahoo Finance 1m data:** Limited to last 7 days
2. **Binance API:** Rate limits apply (weight-based)
3. **SET50 availability:** May have limited 1-minute data

### Trading Considerations

1. **This is a PoC:** No live trading implementation
2. **Transaction costs:** Simplified model (flat 0.1%)
3. **Slippage:** Not modeled
4. **Market impact:** Not considered
5. **Liquidity:** Assumes perfect execution

### Risk Warnings

⚠️ **This is educational software for research purposes only.**

- Past performance does not guarantee future results
- Crypto markets are highly volatile
- Cross-asset relationships can break down
- Always test thoroughly before any real trading

## 🐛 Troubleshooting

### Common Issues

**1. No data fetched:**
- Check internet connection
- Verify API endpoints are accessible
- Try increasing `period` or decreasing `limit`

**2. Correlation errors:**
- Ensure sufficient data points (>100 recommended)
- Check for NaN values in price data
- Verify timestamp alignment

**3. Empty signals:**
- Reduce `window` size if insufficient data
- Adjust `z_entry` threshold (try 1.5)
- Check that returns are being calculated

**4. Import errors:**
- Reinstall requirements: `pip install -r requirements.txt`
- Check Python version (3.8+ required)
- Activate virtual environment

## 📚 References

### Academic Papers
- Granger, C.W.J. (1969). "Investigating Causal Relations by Econometric Models"
- Lo, A.W. & MacKinlay, A.C. (1990). "When Are Contrarian Profits Due to Stock Market Overreaction?"

### Books
- "Algorithmic Trading" by Ernie Chan
- "Quantitative Trading" by Ernie Chan
- "Trading and Exchanges" by Larry Harris

### APIs Used
- Binance API: https://binance-docs.github.io/apidocs/spot/en/
- Yahoo Finance API: https://pypi.org/project/yfinance/

## 📄 License

This project is provided as-is for educational and research purposes.

## 👥 Authors

Quantitative Development Team - Lead-Lag POC Project

## 📧 Contact

For questions or feedback, please open an issue in the repository.

---

**Last Updated:** October 2025
**Version:** 1.0.0 (PoC)
