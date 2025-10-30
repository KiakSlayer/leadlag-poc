"""
Data Fetcher Module for Cross-Asset Lead-Lag System
====================================================
Fetches 1-minute OHLCV data from multiple sources:
- Crypto: Binance API (BTCUSDT, ETHUSDT, SOLUSDT)
- Equity Indices: Yahoo Finance (S&P500, NASDAQ)
- SET50: Yahoo Finance fallback (^SET.BK)
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time
import yfinance as yf


class DataFetcher:
    """Unified data fetcher for crypto and equity indices"""

    def __init__(self):
        self.binance_base = "https://api.binance.com"
        self.cache = {}

    def fetch_crypto_ohlcv(
        self,
        symbol: str,
        interval: str = "1m",
        limit: int = 1000,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch crypto OHLCV data from Binance

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Timeframe ('1m', '5m', '15m', '1h', '1d')
            limit: Number of candles (max 1000)
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        url = f"{self.binance_base}/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }

        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades',
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])

            # Convert types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Keep only essential columns
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df.set_index('timestamp', inplace=True)

            print(f"✓ Fetched {len(df)} {interval} bars for {symbol}")
            return df

        except requests.exceptions.RequestException as e:
            print(f"✗ Error fetching {symbol} from Binance: {e}")
            return pd.DataFrame()

    def fetch_equity_ohlcv(
        self,
        symbol: str,
        period: str = "7d",
        interval: str = "1m"
    ) -> pd.DataFrame:
        """
        Fetch equity/index data from Yahoo Finance

        Args:
            symbol: Ticker symbol (e.g., '^GSPC' for S&P500, '^IXIC' for NASDAQ)
            period: Data period ('1d', '5d', '1mo', '3mo', '1y')
            interval: Timeframe ('1m', '5m', '15m', '1h', '1d')

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume

        Note:
            Yahoo Finance 1m data is limited to last 7 days
        """
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)

            if df.empty:
                print(f"✗ No data returned for {symbol}")
                return pd.DataFrame()

            # Reset index first to get datetime as a column
            df.reset_index(inplace=True)

            # Standardize column names to lowercase
            df.columns = df.columns.str.lower()

            # The datetime column could be named 'date', 'datetime', or 'index'
            # Find it and rename to 'timestamp'
            datetime_col = None
            for col in df.columns:
                if col in ['date', 'datetime', 'index']:
                    datetime_col = col
                    break

            if datetime_col:
                df = df.rename(columns={datetime_col: 'timestamp'})
            elif 'timestamp' not in df.columns:
                # If still no timestamp column found, assume first column is datetime
                df = df.rename(columns={df.columns[0]: 'timestamp'})

            # Ensure timestamp is datetime type
            if 'timestamp' in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            else:
                print(f"✗ Could not identify timestamp column for {symbol}")
                return pd.DataFrame()

            # Keep only essential columns
            cols_to_keep = ['open', 'high', 'low', 'close', 'volume']
            df = df[[col for col in cols_to_keep if col in df.columns]]

            print(f"✓ Fetched {len(df)} {interval} bars for {symbol}")
            return df

        except Exception as e:
            print(f"✗ Error fetching {symbol} from Yahoo Finance: {e}")
            return pd.DataFrame()

    def fetch_set50_ohlcv(
        self,
        period: str = "7d",
        interval: str = "1m"
    ) -> pd.DataFrame:
        """
        Fetch SET50 index data (Thailand)
        Using ^SET.BK as proxy via Yahoo Finance

        Args:
            period: Data period
            interval: Timeframe

        Returns:
            DataFrame with OHLCV data
        """
        # SET50 ticker on Yahoo Finance
        # ^SET.BK is SET Index, SET50 might not have 1m data
        # Using SET Index as proxy
        return self.fetch_equity_ohlcv("^SET.BK", period=period, interval=interval)

    def fetch_all_assets(
        self,
        crypto_symbols: List[str] = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'],
        equity_symbols: Dict[str, str] = {
            'SP500': '^GSPC',
            'NASDAQ': '^IXIC',
            'SET50': '^SET.BK'
        },
        period: str = "7d",
        interval: str = "1m"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch all assets data in one call

        Args:
            crypto_symbols: List of crypto pairs
            equity_symbols: Dict of name -> ticker mapping
            period: Period for equity data
            interval: Timeframe

        Returns:
            Dictionary of {asset_name: DataFrame}
        """
        all_data = {}

        # Fetch crypto data
        print("\n📊 Fetching Crypto Data...")
        for symbol in crypto_symbols:
            df = self.fetch_crypto_ohlcv(symbol, interval=interval, limit=1000)
            if not df.empty:
                all_data[symbol] = df
            time.sleep(0.5)  # Rate limiting

        # Fetch equity data
        print("\n📈 Fetching Equity Index Data...")
        for name, ticker in equity_symbols.items():
            df = self.fetch_equity_ohlcv(ticker, period=period, interval=interval)
            if not df.empty:
                all_data[name] = df
            time.sleep(0.5)  # Rate limiting

        print(f"\n✓ Successfully fetched data for {len(all_data)} assets")
        return all_data

    def align_timestamps(
        self,
        data_dict: Dict[str, pd.DataFrame],
        method: str = "inner",
        tolerance: str = "1min"
    ) -> Dict[str, pd.DataFrame]:
        """
        Align all dataframes to common timestamps with timezone and frequency normalization

        Args:
            data_dict: Dictionary of DataFrames
            method: Merge method ('inner' for intersection, 'outer' for union)
            tolerance: Time tolerance for alignment (default: '1min')

        Returns:
            Dictionary with aligned DataFrames
        """
        if not data_dict:
            return {}

        # Step 1: Normalize all timestamps (remove timezone, round to minute)
        normalized_data = {}
        for name, df in data_dict.items():
            df_copy = df.copy()

            # Convert to timezone-naive UTC
            if df_copy.index.tz is not None:
                df_copy.index = df_copy.index.tz_convert('UTC').tz_localize(None)

            # Round to nearest minute for alignment
            df_copy.index = df_copy.index.round('1min')

            # Remove any duplicate timestamps (keep first)
            df_copy = df_copy[~df_copy.index.duplicated(keep='first')]

            normalized_data[name] = df_copy

        # Step 2: Find common timestamp range
        all_indices = [df.index for df in normalized_data.values()]

        if method == "inner":
            # Use intersection of all timestamps
            common_index = all_indices[0]
            for idx in all_indices[1:]:
                common_index = common_index.intersection(idx)

            # If intersection is empty, try outer join with warning
            if len(common_index) == 0:
                print("⚠️  Warning: No exact timestamp overlap found.")
                print("   Switching to 'outer' join with forward fill...")
                print(f"   Asset date ranges:")
                for name, df in normalized_data.items():
                    print(f"     {name:15s}: {df.index.min()} to {df.index.max()} ({len(df)} points)")

                # Fall back to outer join
                common_index = all_indices[0]
                for idx in all_indices[1:]:
                    common_index = common_index.union(idx)
                common_index = common_index.sort_values()
        else:
            # Use union of all timestamps
            common_index = all_indices[0]
            for idx in all_indices[1:]:
                common_index = common_index.union(idx)
            common_index = common_index.sort_values()

        # Step 3: Reindex all dataframes
        aligned_data = {}
        for name, df in normalized_data.items():
            # Use forward fill to handle missing values
            aligned_df = df.reindex(common_index, method='ffill', limit=5)

            # Drop rows with too many NaN values (more than 50% of columns)
            min_non_na = int(np.ceil(len(aligned_df.columns) * 0.5)) if len(aligned_df.columns) else 0
            if min_non_na > 0:
                aligned_df = aligned_df.dropna(thresh=min_non_na)

            aligned_data[name] = aligned_df

        # Step 4: Find common valid timestamps across all assets
        # (timestamps where all assets have data)
        if len(aligned_data) > 1:
            valid_indices = None
            for name, df in aligned_data.items():
                valid_idx = df.dropna().index
                if valid_indices is None:
                    valid_indices = valid_idx
                else:
                    valid_indices = valid_indices.intersection(valid_idx)

            # Filter all dataframes to only common valid timestamps
            if valid_indices is not None and len(valid_indices) > 0:
                aligned_data = {
                    name: df.loc[valid_indices]
                    for name, df in aligned_data.items()
                }
                common_index = valid_indices

        print(f"✓ Aligned data to {len(common_index)} common timestamps")

        if len(common_index) < 10:
            print(f"⚠️  Warning: Only {len(common_index)} common timestamps found!")
            print("   Consider using longer time period or different interval")

        return aligned_data

    def get_close_prices(
        self,
        data_dict: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Extract close prices from all assets into single DataFrame

        Args:
            data_dict: Dictionary of DataFrames

        Returns:
            DataFrame with close prices for all assets
        """
        close_prices = {}
        for name, df in data_dict.items():
            if 'close' in df.columns:
                close_prices[name] = df['close']

        prices_df = pd.DataFrame(close_prices)
        return prices_df


if __name__ == "__main__":
    # Test the data fetcher
    fetcher = DataFetcher()

    print("=" * 60)
    print("Testing Cross-Asset Data Fetcher")
    print("=" * 60)

    # Fetch all data
    data = fetcher.fetch_all_assets(
        crypto_symbols=['BTCUSDT', 'ETHUSDT', 'SOLUSDT'],
        equity_symbols={'SP500': '^GSPC', 'NASDAQ': '^IXIC'},
        period="5d",
        interval="1m"
    )

    # Align timestamps
    aligned_data = fetcher.align_timestamps(data, method="inner")

    # Get close prices
    prices = fetcher.get_close_prices(aligned_data)

    print("\n" + "=" * 60)
    print("Close Prices DataFrame:")
    print("=" * 60)
    print(prices.head())
    print(f"\nShape: {prices.shape}")
    print(f"Date range: {prices.index.min()} to {prices.index.max()}")
