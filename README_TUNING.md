# Parameter Tuning System - Cross-Asset Lead-Lag Strategy

## Overview

The **Parameter Tuning System** systematically tests hyperparameter combinations to identify optimal and robust configurations for the lead-lag arbitrage strategy.

## 📁 Files

### Core Scripts
- **`tuner.py`** - Main parameter sweep engine
- **`tuning_analysis.ipynb`** - Jupyter notebook for visualization and analysis

### Generated Outputs
- **`tuning_results.csv`** - Complete results of all tested configurations
- **`top_configurations.csv`** - Top 10 most robust configurations
- **`tuning_summary.csv`** - Summary statistics and recommendations

### Visualizations
- **`sensitivity_analysis.png`** - Parameter-wise impact charts
- **`heatmaps_sharpe.png`** - Parameter interaction heatmaps
- **`heatmaps_multi_metric.png`** - Multi-metric performance heatmaps
- **`robustness_analysis.png`** - Robustness score analysis
- **`performance_by_category.png`** - Performance by pair category

## 🚀 Quick Start

### 1. Run Parameter Sweep

```bash
# Basic usage (default parameters)
python tuner.py

# Custom configuration
python tuner.py \
  --intervals 5m 15m 30m \
  --windows 30 60 90 \
  --z-entries 1.5 2.0 2.5 \
  --z-exits 0.3 0.5 \
  --period 7d \
  --output my_results.csv
```

**Default Parameter Space:**
- Intervals: `5m, 15m`
- Windows: `30, 60, 90`
- Z-Entry: `1.5, 2.0, 2.5`
- Z-Exit: `0.3, 0.5`

**Total Combinations:** 2 × 3 × 3 × 2 = **36 base configurations**

### 2. Analyze Results

```bash
# Open Jupyter notebook
jupyter notebook tuning_analysis.ipynb

# Or convert to HTML report
jupyter nbconvert --to html --execute tuning_analysis.ipynb
```

## 📊 Metrics Recorded

For each configuration, the system records:

| Metric | Description |
|--------|-------------|
| **Sharpe Ratio** | Risk-adjusted returns (primary metric) |
| **Total Return %** | Absolute percentage return |
| **Max Drawdown %** | Maximum peak-to-trough decline |
| **Win Rate** | Percentage of profitable trades |
| **Number of Trades** | Total trades executed |
| **Sortino Ratio** | Downside risk-adjusted returns |
| **Volatility** | Annualized volatility |

## 🎯 Robustness Score

**Formula:** `Robustness = mean(Sharpe) - 0.5 × std(Sharpe)`

This metric:
- ✅ Rewards high average performance
- ⚠️ Penalizes high variance across different pairs/conditions
- 🎯 Identifies configurations that work consistently

## 📈 Analysis Workflow

The Jupyter notebook performs:

1. **Load Results** - Import CSV data
2. **Parameter Sensitivity** - Analyze each parameter's impact
3. **Interaction Heatmaps** - Visualize parameter combinations
4. **Multi-Metric Analysis** - Compare across different metrics
5. **Robustness Analysis** - Calculate stability scores
6. **Top 5 Ranking** - Identify best configurations
7. **Category Analysis** - Compare pair types
8. **Recommendations** - Generate actionable insights

## 🔧 Command-Line Options

### Data Options
```bash
--crypto BTCUSDT ETHUSDT SOLUSDT    # Crypto symbols to test
--period 7d                          # Data period (1d, 5d, 7d, 1mo)
```

### Parameter Space
```bash
--intervals 5m 15m 30m               # Time intervals
--windows 30 60 90 120               # Rolling windows
--z-entries 1.5 2.0 2.5 3.0         # Entry thresholds
--z-exits 0.3 0.5 0.7               # Exit thresholds
```

### Backtest Config
```bash
--capital 100000                     # Initial capital
--cost 0.001                         # Transaction cost (0.1%)
--max-pairs 3                        # Max pairs per interval
```

### Output
```bash
--output custom_results.csv          # Output filename
```

## 📋 Example Output Structure

### tuning_results.csv
```csv
interval,window,z_entry,z_exit,pair_category,leader,lagger,lag,sharpe_ratio,total_return_pct,max_drawdown_pct,win_rate,num_trades
5m,60,2.0,0.5,Crypto-Index,BTCUSDT,SP500,2,1.85,12.45,-5.23,0.57,42
15m,90,1.5,0.3,Crypto-Crypto,BTCUSDT,ETHUSDT,1,2.12,18.30,-3.89,0.62,28
...
```

### top_configurations.csv
```csv
interval,window,z_entry,z_exit,sharpe_mean,sharpe_std,robustness_score,return_mean,drawdown_mean,winrate_mean
15m,60,2.0,0.5,1.95,0.35,1.78,15.2,-4.5,0.58
5m,90,2.0,0.3,1.88,0.42,1.67,14.8,-5.1,0.55
...
```

## 📊 Visualization Examples

### 1. Parameter Sensitivity Analysis
Shows average Sharpe Ratio for each parameter value:
- Which interval performs best?
- Optimal window size?
- Best Z-score thresholds?

### 2. Interaction Heatmaps
Reveals synergies between parameters:
- Window × Z-Entry combinations
- Interval × Window performance
- Z-Entry × Z-Exit interactions

### 3. Robustness Analysis
Compares consistency vs. peak performance:
- High Sharpe but unstable → risky
- Moderate Sharpe but consistent → reliable

### 4. Multi-Metric Dashboard
Evaluates trade-offs:
- High return but high drawdown?
- High win rate but few trades?
- Balanced configurations?

## 💡 Interpretation Guide

### Good Configuration Indicators
✅ **Sharpe > 1.5** - Strong risk-adjusted returns  
✅ **Robustness > 1.0** - Consistent performance  
✅ **Max DD < -10%** - Acceptable risk  
✅ **Win Rate > 50%** - Edge in the market  
✅ **Trades > 20** - Sufficient sample size  

### Warning Signs
⚠️ **High Sharpe, High Std** - Overfitting or luck  
⚠️ **Few Trades** - Insufficient validation  
⚠️ **Large Drawdowns** - Unacceptable risk  
⚠️ **Low Robustness** - Unreliable across pairs  

## 🔄 Recommended Workflow

### Phase 1: Initial Sweep
```bash
# Quick test with default parameters
python tuner.py --period 5d --output initial_sweep.csv
```

### Phase 2: Analysis
```bash
# Analyze results
jupyter notebook tuning_analysis.ipynb
```

### Phase 3: Refinement
```bash
# Zoom in on promising regions
python tuner.py \
  --windows 50 55 60 65 70 \
  --z-entries 1.8 1.9 2.0 2.1 2.2 \
  --period 7d \
  --output refined_sweep.csv
```

### Phase 4: Validation
```bash
# Test on different time period
python tuner.py \
  --period 1mo \
  --intervals 15m \
  --windows 60 \
  --z-entries 2.0 \
  --z-exits 0.5 \
  --output validation_test.csv
```

## 🎯 Typical Recommendations

Based on historical testing, common optimal configurations:

**Conservative (Stable):**
```python
interval = '15m'
window = 90
z_entry = 2.5
z_exit = 0.5
```
- Higher thresholds → fewer but higher-confidence trades
- Larger window → more stable statistics

**Aggressive (High Frequency):**
```python
interval = '5m'
window = 30
z_entry = 1.5
z_exit = 0.3
```
- Lower thresholds → more frequent trades
- Smaller window → faster adaptation

**Balanced (Recommended):**
```python
interval = '15m'
window = 60
z_entry = 2.0
z_exit = 0.5
```
- Moderate frequency
- Good balance of responsiveness and stability

## 🧪 Testing Checklist

- [x] **CSV Auto-Generated** - Results saved to `tuning_results.csv`
- [x] **Heatmaps Produced** - Multiple visualization files created
- [x] **Clear Recommendation** - Best config identified with reasoning
- [x] **Robustness Calculation** - Stability metrics computed
- [x] **Top 5 Ranking** - Best configurations ranked and explained
- [x] **Multi-Metric Analysis** - Comprehensive performance evaluation

## ⚡ Performance Tips

### Speed Optimization
1. **Reduce parameter space** - Start broad, then refine
2. **Limit pairs** - Use `--max-pairs 2` for faster testing
3. **Shorter period** - Use `--period 5d` for initial sweeps
4. **Parallel processing** - Future enhancement (Phase 2)

### Memory Management
- Each configuration × pair generates ~1MB of data
- 1000 combinations ≈ 1GB RAM usage
- Clear memory between intervals if needed

## 🚧 Limitations & Considerations

### Data Constraints
- Yahoo Finance 1m data limited to 7 days
- Binance rate limits may slow down data fetching
- Market hours affect equity data availability

### Statistical Validity
- Minimum 100 data points required per configuration
- More trades → more reliable metrics
- Consider multiple time periods for validation

### Overfitting Risk
- Don't tune on same data you'll trade on
- Use walk-forward validation
- Monitor out-of-sample performance

## 📚 Next Steps

After identifying optimal parameters:

1. **Validate** on separate time period
2. **Paper Trade** with recommended config
3. **Monitor** real-time performance
4. **Re-tune** periodically (monthly/quarterly)
5. **Adapt** based on market regime changes

## 🤝 Integration

The tuning system integrates seamlessly with:
- `main_crossasset_poc.py` - Use recommended params
- `core/backtester.py` - Leverage same backtest engine
- Monitoring systems - Track drift from tuned performance

## 📞 Support

For issues or questions:
1. Check output CSVs for data quality
2. Review error messages in tuner logs
3. Validate data availability for chosen periods
4. Adjust parameter ranges if no results

---

**Version:** 1.0  
**Last Updated:** November 2025  
**Status:** Production Ready ✅
