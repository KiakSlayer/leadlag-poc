# Quick Start Guide

## ✅ What's Been Fixed

The **timestamp column handling bug** in `core/data_fetcher.py` has been fixed! The error:
```
✗ Error fetching ^GSPC from Yahoo Finance: 'timestamp'
```

is now resolved.

## 🚀 Getting Started (3 Options)

### Option 1: Crypto-Only Analysis (Works Now!)

If you have yfinance installation issues, analyze crypto pairs only:

```python
from core.data_fetcher import DataFetcher
from core.correlation_analyzer import CorrelationAnalyzer
from core.crossasset_leadlag_model import CrossAssetLeadLagModel, ModelConfig
from core.backtester import Backtester, BacktestConfig
from core.visualizer import StrategyVisualizer

# Fetch crypto data only (works without yfinance)
fetcher = DataFetcher()
data = fetcher.fetch_all_assets(
    crypto_symbols=['BTCUSDT', 'ETHUSDT', 'SOLUSDT'],
    equity_symbols={}  # Skip equity indices
)

# Analyze crypto-crypto pairs
prices = fetcher.get_close_prices(fetcher.align_timestamps(data))

analyzer = CorrelationAnalyzer(prices)
pairs = analyzer.find_best_pairs(
    crypto_assets=['BTCUSDT', 'ETHUSDT', 'SOLUSDT'],
    index_assets=[],
    min_correlation=0.5
)

# Run strategy on BTC-ETH pair
config = ModelConfig(window=60, z_entry=2.0, z_exit=0.5)
model = CrossAssetLeadLagModel(config)
signals = model.run_strategy(prices, 'BTCUSDT', 'ETHUSDT', lag=0)

# Backtest
bt_config = BacktestConfig(initial_capital=100000)
backtester = Backtester(bt_config)
results = backtester.run_backtest(signals, prices, 'BTCUSDT', 'ETHUSDT')

print(results['metrics'])
```

### Option 2: Install yfinance (For Full Functionality)

If yfinance has issues, try these methods in order:

**Method 2a: Use conda (Recommended)**
```bash
conda install -c conda-forge yfinance
```

**Method 2b: Install from source**
```bash
pip install git+https://github.com/ranaroussi/yfinance.git
```

**Method 2c: Skip problematic dependency**
```bash
pip install yfinance --no-deps
pip install pandas requests beautifulsoup4 lxml
```

Then test:
```bash
python -c "import yfinance; print('✓ yfinance installed')"
```

### Option 3: Provide Your Own Data

Use CSV files for equity indices:

```python
import pandas as pd
from core.crossasset_leadlag_model import CrossAssetLeadLagModel, ModelConfig

# Load your own data
btc_data = pd.read_csv('btc_1min.csv', index_col='timestamp', parse_dates=True)
sp500_data = pd.read_csv('sp500_1min.csv', index_col='timestamp', parse_dates=True)

# Combine into single DataFrame
prices = pd.DataFrame({
    'BTCUSDT': btc_data['close'],
    'SP500': sp500_data['close']
})

# Run analysis as usual
config = ModelConfig(window=60, z_entry=2.0, z_exit=0.5)
model = CrossAssetLeadLagModel(config)
signals = model.run_strategy(prices, 'BTCUSDT', 'SP500', lag=0)
```

## 🧪 Test the Fix

Verify the timestamp fix works:

```bash
python legacy/test_data_fetcher_fix.py
```

Expected output:
```
============================================================
✓ SUCCESS! The fix handles timestamp column correctly
============================================================
All tests passed! ✓
```

## 📊 Full Example (Crypto + Equity)

Once yfinance is installed:

```bash
# Run complete analysis
python main_crossasset_poc.py

# Or with custom parameters
python main_crossasset_poc.py \
    --crypto BTCUSDT ETHUSDT \
    --period 5d \
    --interval 1m \
    --window 60 \
    --z-entry 2.0 \
    --z-exit 0.5 \
    --capital 100000
```

## 🔍 What the Fix Does

**Before (Broken):**
```python
df.columns = df.columns.str.lower()  # Lowercase first
df.reset_index(inplace=True)         # Then reset
# ❌ 'timestamp' column doesn't exist yet!
df['timestamp'] = ...  # KeyError!
```

**After (Fixed):**
```python
df.reset_index(inplace=True)         # Reset FIRST
df.columns = df.columns.str.lower()  # Then lowercase
# ✅ Detect datetime column by checking for 'date', 'datetime', 'index'
# ✅ Rename to 'timestamp'
# ✅ Fallback if not found
```

## 📁 New Files

- ✅ `core/data_fetcher.py` - Fixed timestamp handling
- ✅ `legacy/test_data_fetcher_fix.py` - Verification test
- ✅ `INSTALLATION_NOTES.md` - Detailed installation help
- ✅ `QUICKSTART.md` - This file

## 🆘 Troubleshooting

**Error: ModuleNotFoundError: No module named 'yfinance'**
→ Use Option 1 (Crypto-only) or Option 3 (CSV files)

**Error: No data returned for ^GSPC**
→ Yahoo Finance may be rate-limiting. Wait a few minutes and retry.

**Error: Could not identify timestamp column**
→ This should be fixed now. If you still see this, please report the yfinance version:
```bash
python -c "import yfinance; print(yfinance.__version__)"
```

## 📈 Next Steps

1. **Test crypto-only first** to verify everything works
2. **Install yfinance** if you need equity indices
3. **Run full analysis** with `main_crossasset_poc.py`
4. **Customize parameters** to fit your strategy
5. **Integrate with Kafka** for real-time streaming (future)

## 📚 Documentation

- `README_CROSSASSET.md` - Complete system documentation
- `INSTALLATION_NOTES.md` - Installation troubleshooting
- Inline code documentation in all `.py` files

## ✅ Verification Checklist

- [x] Core code fixed (`core/data_fetcher.py`)
- [x] Test created (`legacy/test_data_fetcher_fix.py`)
- [x] Documentation updated
- [x] Alternative approaches provided
- [x] Changes committed and pushed

---

**Status:** 🟢 **READY TO USE**

The timestamp bug is fixed and the system is fully functional!
