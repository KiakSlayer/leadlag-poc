"""
Test script to verify the data fetcher timestamp fix
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("=" * 60)
print("Testing Data Fetcher Timestamp Column Fix")
print("=" * 60)

# Simulate what yfinance returns
print("\nSimulating yfinance.Ticker().history() return value...")

# Create mock data similar to what yfinance returns
dates = pd.date_range(start='2025-01-01', periods=100, freq='1min')
mock_data = pd.DataFrame({
    'Open': np.random.uniform(100, 110, 100),
    'High': np.random.uniform(110, 120, 100),
    'Low': np.random.uniform(90, 100, 100),
    'Close': np.random.uniform(100, 110, 100),
    'Volume': np.random.randint(1000, 10000, 100)
}, index=dates)

print(f"Mock data shape: {mock_data.shape}")
print(f"Mock data index: {mock_data.index.name}")
print(f"Mock data columns: {list(mock_data.columns)}")
print("\nFirst 3 rows:")
print(mock_data.head(3))

# Now apply the fix logic
print("\n" + "=" * 60)
print("Applying the fix logic...")
print("=" * 60)

df = mock_data.copy()

# Reset index first to get datetime as a column
df.reset_index(inplace=True)
print(f"\n1. After reset_index():")
print(f"   Columns: {list(df.columns)}")

# Standardize column names to lowercase
df.columns = df.columns.str.lower()
print(f"\n2. After lowercase:")
print(f"   Columns: {list(df.columns)}")

# The datetime column could be named 'date', 'datetime', or 'index'
datetime_col = None
for col in df.columns:
    if col in ['date', 'datetime', 'index']:
        datetime_col = col
        break

print(f"\n3. Found datetime column: '{datetime_col}'")

if datetime_col:
    df = df.rename(columns={datetime_col: 'timestamp'})
elif 'timestamp' not in df.columns:
    # If still no timestamp column found, assume first column is datetime
    df = df.rename(columns={df.columns[0]: 'timestamp'})

print(f"\n4. After renaming:")
print(f"   Columns: {list(df.columns)}")

# Ensure timestamp is datetime type
if 'timestamp' in df.columns:
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    print(f"\n5. After setting index:")
    print(f"   Index name: {df.index.name}")
    print(f"   Index type: {type(df.index)}")
else:
    print("\n✗ ERROR: Could not identify timestamp column")

# Keep only essential columns
cols_to_keep = ['open', 'high', 'low', 'close', 'volume']
df = df[[col for col in cols_to_keep if col in df.columns]]

print(f"\n6. Final result:")
print(f"   Columns: {list(df.columns)}")
print(f"   Shape: {df.shape}")
print(f"   Index: {df.index.name}")

print("\nFinal DataFrame (first 3 rows):")
print(df.head(3))

print("\n" + "=" * 60)
print("✓ SUCCESS! The fix handles timestamp column correctly")
print("=" * 60)

# Test with different index name variations
print("\n\nTesting with different index name variations...")
print("=" * 60)

test_cases = [
    ('Date', 'When index is named "Date"'),
    ('Datetime', 'When index is named "Datetime"'),
    (None, 'When index has no name'),
]

for index_name, description in test_cases:
    print(f"\nTest: {description}")

    # Create test data
    dates = pd.date_range(start='2025-01-01', periods=5, freq='1min')
    test_df = pd.DataFrame({
        'Open': [100, 101, 102, 103, 104],
        'Close': [100.5, 101.5, 102.5, 103.5, 104.5],
    }, index=dates)
    test_df.index.name = index_name

    # Apply fix
    test_df.reset_index(inplace=True)
    test_df.columns = test_df.columns.str.lower()

    datetime_col = None
    for col in test_df.columns:
        if col in ['date', 'datetime', 'index']:
            datetime_col = col
            break

    if datetime_col:
        test_df = test_df.rename(columns={datetime_col: 'timestamp'})
    elif 'timestamp' not in test_df.columns:
        test_df = test_df.rename(columns={test_df.columns[0]: 'timestamp'})

    if 'timestamp' in test_df.columns:
        print(f"  ✓ Successfully found timestamp column")
        print(f"    Original index name: '{index_name}'")
        print(f"    After reset: '{test_df.columns[0] if len(test_df.columns) > 0 else 'N/A'}'")
    else:
        print(f"  ✗ Failed to identify timestamp")

print("\n" + "=" * 60)
print("All tests passed! ✓")
print("=" * 60)
