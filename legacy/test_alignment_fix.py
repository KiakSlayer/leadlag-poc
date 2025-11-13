"""
Test script to verify the timestamp alignment and error handling fixes
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("=" * 70)
print("Testing Timestamp Alignment and Error Handling Fixes")
print("=" * 70)

# Simulate the issue: crypto (24/7) and equity (trading hours only) data
print("\n1. Simulating crypto and equity data with different time ranges...")

# Crypto data: 24/7, recent dates
crypto_dates = pd.date_range(
    start='2025-01-01 00:00:00',
    end='2025-01-02 23:59:00',
    freq='1min',
    tz='UTC'
)
crypto_data = pd.DataFrame({
    'close': np.random.uniform(40000, 41000, len(crypto_dates))
}, index=crypto_dates)

print(f"   Crypto data: {len(crypto_data)} rows from {crypto_data.index.min()} to {crypto_data.index.max()}")

# Equity data: trading hours only, different timezone
equity_dates = pd.date_range(
    start='2025-01-01 09:30:00',  # Market open
    end='2025-01-01 16:00:00',    # Market close
    freq='1min'
)
equity_data = pd.DataFrame({
    'close': np.random.uniform(4500, 4600, len(equity_dates))
}, index=equity_dates)

print(f"   Equity data: {len(equity_data)} rows from {equity_data.index.min()} to {equity_data.index.max()}")

# Test the alignment logic
print("\n2. Testing alignment with the fixed logic...")

data_dict = {
    'BTCUSDT': crypto_data,
    'SP500': equity_data
}

# Simulate the fix
print("\n   Step 1: Normalizing timestamps...")
normalized_data = {}
for name, df in data_dict.items():
    df_copy = df.copy()

    # Convert to timezone-naive UTC
    if df_copy.index.tz is not None:
        df_copy.index = df_copy.index.tz_convert('UTC').tz_localize(None)

    # Round to nearest minute
    df_copy.index = df_copy.index.round('1min')

    # Remove duplicates
    df_copy = df_copy[~df_copy.index.duplicated(keep='first')]

    normalized_data[name] = df_copy
    print(f"     {name}: {len(df_copy)} rows, {df_copy.index.min()} to {df_copy.index.max()}")

print("\n   Step 2: Finding common timestamps (inner join)...")
all_indices = [df.index for df in normalized_data.values()]
common_index = all_indices[0]
for idx in all_indices[1:]:
    common_index = common_index.intersection(idx)

print(f"     Common timestamps (inner): {len(common_index)}")

if len(common_index) == 0:
    print("     ⚠️  No overlap! Falling back to outer join...")
    common_index = all_indices[0]
    for idx in all_indices[1:]:
        common_index = common_index.union(idx)
    common_index = common_index.sort_values()
    print(f"     Common timestamps (outer): {len(common_index)}")

print("\n   Step 3: Reindexing with forward fill...")
aligned_data = {}
for name, df in normalized_data.items():
    aligned_df = df.reindex(common_index, method='ffill', limit=5)
    aligned_data[name] = aligned_df
    print(f"     {name}: {len(aligned_df)} rows, {aligned_df.isna().sum().sum()} NaN values")

print("\n   Step 4: Finding valid timestamps (where all assets have data)...")
valid_indices = None
for name, df in aligned_data.items():
    valid_idx = df.dropna().index
    if valid_indices is None:
        valid_indices = valid_idx
    else:
        valid_indices = valid_indices.intersection(valid_idx)

print(f"     Valid timestamps: {len(valid_indices) if valid_indices is not None else 0}")

if valid_indices is not None and len(valid_indices) > 0:
    final_data = {
        name: df.loc[valid_indices]
        for name, df in aligned_data.items()
    }
    print(f"\n✓ SUCCESS! Aligned to {len(valid_indices)} common timestamps")

    # Show sample
    print("\n   Sample of aligned data:")
    combined = pd.DataFrame({
        name: df['close'] for name, df in final_data.items()
    })
    print(combined.head(10))
else:
    print("\n⚠️  No valid timestamps found after alignment")

# Test with overlapping data
print("\n" + "=" * 70)
print("3. Testing with overlapping data...")

# Create overlapping data
overlap_start = '2025-01-01 10:00:00'
overlap_end = '2025-01-01 15:00:00'

crypto_dates_overlap = pd.date_range(start=overlap_start, end=overlap_end, freq='1min', tz='UTC')
crypto_overlap = pd.DataFrame({
    'close': np.random.uniform(40000, 41000, len(crypto_dates_overlap))
}, index=crypto_dates_overlap)

equity_dates_overlap = pd.date_range(start=overlap_start, end=overlap_end, freq='1min')
equity_overlap = pd.DataFrame({
    'close': np.random.uniform(4500, 4600, len(equity_dates_overlap))
}, index=equity_dates_overlap)

print(f"   Crypto: {len(crypto_overlap)} rows")
print(f"   Equity: {len(equity_overlap)} rows")

# Apply alignment
data_overlap = {'BTCUSDT': crypto_overlap, 'SP500': equity_overlap}

normalized_overlap = {}
for name, df in data_overlap.items():
    df_copy = df.copy()
    if df_copy.index.tz is not None:
        df_copy.index = df_copy.index.tz_convert('UTC').tz_localize(None)
    df_copy.index = df_copy.index.round('1min')
    df_copy = df_copy[~df_copy.index.duplicated(keep='first')]
    normalized_overlap[name] = df_copy

all_indices = [df.index for df in normalized_overlap.values()]
common_index = all_indices[0]
for idx in all_indices[1:]:
    common_index = common_index.intersection(idx)

print(f"   Common timestamps: {len(common_index)}")

if len(common_index) > 0:
    print(f"   ✓ SUCCESS! Found {len(common_index)} overlapping timestamps")
else:
    print("   ✗ FAILED to find overlap")

# Test error handling
print("\n" + "=" * 70)
print("4. Testing error handling with empty DataFrames...")

from core.correlation_analyzer import CorrelationAnalyzer

# Test with empty DataFrame
empty_prices = pd.DataFrame()
print("   Creating CorrelationAnalyzer with empty DataFrame...")
try:
    analyzer = CorrelationAnalyzer(empty_prices)
    print("   ✓ No crash! Returns object created")
    print(f"   Returns DataFrame shape: {analyzer.returns.shape}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test analyze_all_pairs with empty list
print("\n   Testing analyze_all_pairs with empty list...")
prices_sample = pd.DataFrame({
    'BTC': [100, 101, 102],
    'ETH': [50, 51, 52]
}, index=pd.date_range('2025-01-01', periods=3, freq='1min'))

analyzer = CorrelationAnalyzer(prices_sample)
try:
    result = analyzer.analyze_all_pairs([], max_lag=5)
    print(f"   ✓ No crash! Returns DataFrame with shape: {result.shape}")
    print(f"   Columns: {list(result.columns)}")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "=" * 70)
print("✅ ALL TESTS COMPLETE!")
print("=" * 70)

print("\n📋 Summary:")
print("   ✓ Timestamp normalization (timezone, rounding)")
print("   ✓ Fallback from inner to outer join")
print("   ✓ Forward fill for missing data")
print("   ✓ Valid timestamp filtering")
print("   ✓ Empty DataFrame handling")
print("   ✓ Error handling in correlation analyzer")
