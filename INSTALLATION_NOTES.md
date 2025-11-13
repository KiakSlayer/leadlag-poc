# Installation Notes for Cross-Asset Lead-Lag System

## Quick Install (Recommended)

```bash
pip install pandas numpy requests matplotlib scipy ccxt
```

## yfinance Installation Issues

If you encounter errors installing `yfinance` (particularly with the `multitasking` dependency), here are solutions:

### Solution 1: Use Pre-built Binaries (Recommended)

```bash
# Install from conda-forge if using conda
conda install -c conda-forge yfinance

# Or use a pre-compiled wheel
pip install --only-binary :all: yfinance
```

### Solution 2: Install Without yfinance

The system can work without yfinance if you:
1. Only analyze crypto-crypto pairs (BTC-ETH, etc.)
2. Provide your own equity data via CSV files
3. Use alternative data sources

**Run crypto-only analysis:**
```bash
# Modify main_crossasset_poc.py to skip equity indices
python main_crossasset_poc.py --crypto BTCUSDT ETHUSDT SOLUSDT
```

### Solution 3: Manual Installation

```bash
# Install dependencies manually
pip install pandas numpy requests beautifulsoup4 lxml html5lib

# Install multitasking from source
pip install git+https://github.com/ranaroussi/multitasking.git

# Then install yfinance
pip install yfinance
```

### Solution 4: Use Docker

```bash
# Use the docker-compose.yml for a pre-configured environment
docker-compose up -d
```

## Alternative: Use CSV Data Files

If yfinance installation fails, you can provide data via CSV files:

```python
# Example: Load your own equity data
import pandas as pd

# Your CSV should have columns: timestamp, open, high, low, close, volume
sp500_data = pd.read_csv('sp500_1min.csv', parse_dates=['timestamp'], index_col='timestamp')

# Then use directly in the analysis
from crossasset_leadlag_model import CrossAssetLeadLagModel

prices = pd.DataFrame({
    'BTCUSDT': btc_data['close'],
    'SP500': sp500_data['close']
})

# Run model...
```

## Testing Without yfinance

You can test the core functionality without yfinance:

```bash
# Test individual modules
python legacy/test_data_fetcher_fix.py      # Tests the timestamp fix
python core/correlation_analyzer.py       # Requires pandas only
python core/crossasset_leadlag_model.py   # Requires pandas only
python core/backtester.py                 # Requires pandas only
```

## Minimal Installation for Testing

```bash
# Install only core dependencies
pip install pandas numpy matplotlib

# Test with crypto-only data
python -c "
from data_fetcher import DataFetcher
fetcher = DataFetcher()
data = fetcher.fetch_crypto_ohlcv('BTCUSDT', interval='1m', limit=100)
print(data.head())
"
```

## Platform-Specific Notes

### Linux/WSL
- May need to install build tools: `sudo apt-get install build-essential python3-dev`

### macOS
- May need Xcode Command Line Tools: `xcode-select --install`

### Windows
- Install Microsoft Visual C++ 14.0 or greater
- Or use pre-built wheels from https://www.lfd.uci.edu/~gohlke/pythonlibs/

## Verification

Test your installation:

```bash
python -c "import pandas, numpy, matplotlib; print('✓ Core dependencies OK')"
python -c "import yfinance; print('✓ yfinance OK')"
python legacy/test_data_fetcher_fix.py
```

## Support

If issues persist:
1. Check Python version: `python --version` (requires 3.8+)
2. Upgrade pip: `pip install --upgrade pip setuptools wheel`
3. Use virtual environment: `python -m venv .venv && source .venv/bin/activate`
4. Open an issue with error details
