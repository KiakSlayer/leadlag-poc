"""
Data Fetcher Module for Cross-Asset Lead-Lag System
====================================================
Fetches OHLCV data from Yahoo Finance for both crypto and equity indices.

Updated: Now uses Yahoo Finance for ALL assets (crypto + equities)
This provides more historical data points compared to Binance API.

Author: Quantitative Development Team
Date: November 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time
import yfinance as yf


class DataFetcher:
    """Unified data fetcher using Yahoo Finance for all assets"""

    def __init__(self):
        """Initialize data fetcher"""
        self.cache = {}
        
        # Symbol mapping: User-friendly names to Yahoo Finance tickers
        self.crypto_symbol_map = {
            'BTCUSDT': 'BTC-USD',
            'ETHUSDT': 'ETH-USD',
            'SOLUSDT': 'SOL-USD',
            'BNBUSDT': 'BNB-USD',
            'ADAUSDT': 'ADA-USD',
            'XRPUSDT': 'XRP-USD',
            'DOGEUSDT': 'DOGE-USD',
            'DOTUSDT': 'DOT-USD',
            'MATICUSDT': 'MATIC-USD',
            'AVAXUSDT': 'AVAX-USD',
        }

    def _convert_symbol_to_yahoo(self, symbol: str) -> str:
        """
        Convert symbol to Yahoo Finance format
        
        Args:
            symbol: Symbol in Binance format (e.g., 'BTCUSDT')
            
        Returns:
            Yahoo Finance ticker (e.g., 'BTC-USD')
        """
        # If already in Yahoo format, return as-is
        if '-' in symbol or '^' in symbol:
            return symbol
            
        # Check if it's in our mapping
        if symbol in self.crypto_symbol_map:
            return self.crypto_symbol_map[symbol]
            
        # Try to auto-convert XXXUSDT format
        if symbol.endswith('USDT'):
            base = symbol[:-4]  # Remove 'USDT'
            return f"{base}-USD"
            
        # Return as-is if no conversion found
        return symbol

    def fetch_crypto_ohlcv(
        self,
        symbol: str,
        interval: str = "1m",
        limit: int = 1000,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch crypto OHLCV data from Yahoo Finance
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT' or 'BTC-USD')
            interval: Timeframe ('1m', '5m', '15m', '1h', '1d')
            limit: Number of candles (used to calculate period)
            start_time: Start timestamp in milliseconds (optional)
            end_time: End timestamp in milliseconds (optional)
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        # Convert symbol to Yahoo Finance format
        yahoo_symbol = self._convert_symbol_to_yahoo(symbol)
        
        # Calculate period from limit if not using start/end times
        if start_time is None and end_time is None:
            period = self._calculate_period_from_limit(interval, limit)
        else:
            period = None
        
        print(f"  Fetching {symbol} ({yahoo_symbol}) with interval={interval}...")
        
        try:
            ticker = yf.Ticker(yahoo_symbol)
            
            if period:
                df = ticker.history(period=period, interval=interval)
            else:
                # Convert timestamps to datetime
                start = pd.to_datetime(start_time, unit='ms') if start_time else None
                end = pd.to_datetime(end_time, unit='ms') if end_time else None
                df = ticker.history(start=start, end=end, interval=interval)
            
            if df.empty:
                print(f"  ✗ No data returned for {symbol}")
                return pd.DataFrame()
            
            # Reset index to get datetime as column
            df.reset_index(inplace=True)
            
            # Standardize column names
            df.columns = df.columns.str.lower()
            
            # Find datetime column
            datetime_col = None
            for col in ['date', 'datetime', 'index']:
                if col in df.columns:
                    datetime_col = col
                    break
            
            if datetime_col:
                df = df.rename(columns={datetime_col: 'timestamp'})
            else:
                df = df.rename(columns={df.columns[0]: 'timestamp'})
            
            # Ensure timestamp is datetime
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Keep only essential columns
            essential_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            df = df[[col for col in essential_cols if col in df.columns]]
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            print(f"  ✓ Fetched {len(df)} {interval} bars for {symbol}")
            return df
            
        except Exception as e:
            print(f"  ✗ Error fetching {symbol} from Yahoo Finance: {e}")
            return pd.DataFrame()

    def _calculate_period_from_limit(self, interval: str, limit: int) -> str:
        """
        Calculate Yahoo Finance period string from interval and limit
        
        Args:
            interval: Time interval (e.g., '1m', '5m', '1h', '1d')
            limit: Number of candles desired
            
        Returns:
            Period string for Yahoo Finance (e.g., '7d', '1mo', '3mo')
        """
        # Map intervals to approximate days
        interval_to_minutes = {
            '1m': 1,
            '2m': 2,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '90m': 90,
            '1d': 1440,
        }
        
        # Get minutes per candle
        minutes = interval_to_minutes.get(interval, 1)
        
        # Calculate total days needed (assuming ~390 trading minutes per day for stocks)
        # For crypto, use 1440 minutes per day
        minutes_per_day = 1440 if interval in ['1m', '5m', '15m', '30m', '1h'] else 390
        total_days = (limit * minutes) / minutes_per_day
        
        # Map to Yahoo Finance period strings
        if total_days <= 1:
            return '1d'
        elif total_days <= 5:
            return '5d'
        elif total_days <= 7:
            return '7d'
        elif total_days <= 30:
            return '1mo'
        elif total_days <= 90:
            return '3mo'
        elif total_days <= 180:
            return '6mo'
        elif total_days <= 365:
            return '1y'
        elif total_days <= 730:
            return '2y'
        elif total_days <= 1825:
            return '5y'
        else:
            return 'max'

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
                print(f"  ✗ No data returned for {symbol}")
                return pd.DataFrame()
            
            # Reset index to get datetime as column
            df.reset_index(inplace=True)
            
            # Standardize column names
            df.columns = df.columns.str.lower()
            
            # Find datetime column
            datetime_col = None
            for col in ['date', 'datetime', 'index']:
                if col in df.columns:
                    datetime_col = col
                    break
            
            if datetime_col:
                df = df.rename(columns={datetime_col: 'timestamp'})
            else:
                df = df.rename(columns={df.columns[0]: 'timestamp'})
            
            # Ensure timestamp is datetime
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            # Keep only essential columns
            essential_cols = ['open', 'high', 'low', 'close', 'volume']
            df = df[[col for col in essential_cols if col in df.columns]]
            
            print(f"  ✓ Fetched {len(df)} {interval} bars for {symbol}")
            return df
            
        except Exception as e:
            print(f"  ✗ Error fetching {symbol} from Yahoo Finance: {e}")
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
            crypto_symbols: List of crypto pairs (Binance format or Yahoo format)
            equity_symbols: Dict of name -> ticker mapping
            period: Period for data
            interval: Timeframe
            
        Returns:
            Dictionary of {asset_name: DataFrame}
        """
        all_data = {}
        
        # Fetch crypto data (using Yahoo Finance now)
        print("\n📊 Fetching Crypto Data (via Yahoo Finance)...")
        for symbol in crypto_symbols:
            df = self.fetch_crypto_ohlcv(symbol, interval=interval, limit=1000)
            if not df.empty:
                all_data[symbol] = df
            time.sleep(0.5)  # Rate limiting
        
        # Fetch equity data
        print("\n📈 Fetching Equity Index Data (via Yahoo Finance)...")
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
    print("Testing Yahoo Finance Data Fetcher (All Assets)")
    print("=" * 60)
    
    # Test symbol conversion
    print("\n📋 Symbol Conversion Test:")
    test_symbols = ['BTCUSDT', 'ETHUSDT', 'BTC-USD', '^GSPC']
    for sym in test_symbols:
        converted = fetcher._convert_symbol_to_yahoo(sym)
        print(f"  {sym:15s} → {converted}")
    
    # Fetch all data
    print("\n" + "=" * 60)
    data = fetcher.fetch_all_assets(
        crypto_symbols=['BTCUSDT', 'ETHUSDT', 'SOLUSDT'],
        equity_symbols={'SP500': '^GSPC', 'NASDAQ': '^IXIC'},
        period="1mo",  # Using longer period to get more data
        interval="1h"  # Using hourly data for better compatibility
    )
    
    # Align timestamps
    aligned_data = fetcher.align_timestamps(data, method="inner")
    
    # Get close prices
    prices = fetcher.get_close_prices(aligned_data)
    
    print("\n" + "=" * 60)
    print("Close Prices DataFrame:")
    print("=" * 60)
    print(prices.head(10))
    print(f"\nShape: {prices.shape}")
    print(f"Date range: {prices.index.min()} to {prices.index.max()}")
    print("\nData points per asset:")
    for col in prices.columns:
        print(f"  {col:15s}: {prices[col].count()} points")