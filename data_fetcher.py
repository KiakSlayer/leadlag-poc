"""Utilities for collecting market data used throughout the lead-lag workflow."""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from data_cache import DataCache


class DataFetcher:
    """Unified data fetcher for crypto and equity indices."""

    def __init__(self, use_cache: bool = False, cache_root: str = "cache") -> None:
        self.binance_base = "https://api.binance.com"
        self._cache: Optional[DataCache] = DataCache(cache_root) if use_cache else None

    @staticmethod
    def _ensure_utc(ts: pd.Timestamp) -> pd.Timestamp:
        """Return the timestamp as UTC-normalised (tz-aware) value."""

        ts = pd.Timestamp(ts)
        if ts.tzinfo is None:
            return ts.tz_localize("UTC")
        return ts.tz_convert("UTC")

    @staticmethod
    def _interval_to_millis(interval: str) -> int:
        """Convert a Binance interval string to milliseconds."""

        delta = pd.to_timedelta(interval)
        return int(delta.total_seconds() * 1000)

    @staticmethod
    def _parse_crypto_klines(raw_klines: List[List]) -> pd.DataFrame:
        """Convert raw Binance klines payload to a typed OHLCV DataFrame."""

        if not raw_klines:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        frame = pd.DataFrame(
            raw_klines,
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "trades",
                "taker_buy_base",
                "taker_buy_quote",
                "ignore",
            ],
        )

        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
        for column in ["open", "high", "low", "close", "volume"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        frame = frame.set_index("timestamp")[ ["open", "high", "low", "close", "volume"] ]
        frame.index = frame.index.tz_localize(None)
        frame = frame[~frame.index.duplicated(keep="first")]  # de-duplicate
        return frame.sort_index()

    def fetch_crypto_ohlcv(
        self,
        symbol: str,
        interval: str = "1m",
        limit: int = 1000,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
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
            raw = response.json()

            df = self._parse_crypto_klines(raw)

            print(f"✓ Fetched {len(df)} {interval} bars for {symbol}")
            return df

        except requests.exceptions.RequestException as e:
            print(f"✗ Error fetching {symbol} from Binance: {e}")
            return pd.DataFrame()

    def fetch_equity_ohlcv(
        self,
        symbol: str,
        period: str = "7d",
        interval: str = "1m",
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
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
            interval_to_use = interval

            if interval == "1m":
                if start is not None and end is not None:
                    span = pd.Timestamp(end) - pd.Timestamp(start)
                    if span > pd.Timedelta(days=7):
                        print(
                            "⚠️  1m interval exceeds Yahoo Finance 7-day limit; falling back to 5m."
                        )
                        interval_to_use = "5m"
                elif period not in {"1d", "5d", "7d"}:
                    print(
                        "⚠️  1m interval only supports up to 7 days; falling back to 5m."
                    )
                    interval_to_use = "5m"

            ticker = yf.Ticker(symbol)
            if start is not None or end is not None:
                df = ticker.history(start=start, end=end, interval=interval_to_use)
            else:
                df = ticker.history(period=period, interval=interval_to_use)

            if df.empty:
                print(f"✗ No data returned for {symbol}")
                return pd.DataFrame()

            # Reset index first to get datetime as a column
            df.reset_index(inplace=True)

            # Standardize column names to lowercase
            df.columns = df.columns.str.lower()

            datetime_col = None
            for col in df.columns:
                if col in ["date", "datetime", "index"]:
                    datetime_col = col
                    break

            if datetime_col:
                df = df.rename(columns={datetime_col: "timestamp"})
            elif "timestamp" not in df.columns:
                df = df.rename(columns={df.columns[0]: "timestamp"})

            if "timestamp" in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                else:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df.set_index("timestamp", inplace=True)
            else:
                print(f"✗ Could not identify timestamp column for {symbol}")
                return pd.DataFrame()

            cols_to_keep = ["open", "high", "low", "close", "volume"]
            df = df[[col for col in cols_to_keep if col in df.columns]]
            df.index = df.index.tz_localize(None)

            print(f"✓ Fetched {len(df)} {interval_to_use} bars for {symbol}")
            return df.sort_index()

        except Exception as e:
            print(f"✗ Error fetching {symbol} from Yahoo Finance: {e}")
            return pd.DataFrame()

    def fetch_set50_ohlcv(
        self,
        period: str = "7d",
        interval: str = "1m",
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
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
        return self.fetch_equity_ohlcv("^SET.BK", period=period, interval=interval, start=start, end=end)

    def fetch_crypto_ohlcv_range(
        self,
        symbol: str,
        interval: str,
        start_dt: pd.Timestamp,
        end_dt: pd.Timestamp,
        limit_per_call: int = 1000,
        sleep_sec: float = 0.2,
    ) -> pd.DataFrame:
        """Fetch a long-range slice of Binance OHLCV data via pagination."""

        start_utc = self._ensure_utc(start_dt)
        end_utc = self._ensure_utc(end_dt)

        if start_utc >= end_utc:
            print("✗ Invalid date range supplied for crypto fetch.")
            return pd.DataFrame()

        cache_key = {
            "source": "binance",
            "symbol": symbol,
            "interval": interval,
            "start": start_utc.isoformat(),
            "end": end_utc.isoformat(),
        }

        if self._cache is not None:
            cached = self._cache.load(**cache_key)
            if cached is not None and not cached.empty:
                print(
                    f"✓ Cache hit for {symbol} ({interval}) covering {len(cached)} bars"
                )
                return cached

        start_ms = int(start_utc.value // 1_000_000)
        end_ms = int(end_utc.value // 1_000_000)
        step_ms = self._interval_to_millis(interval)
        limit_span = step_ms * limit_per_call

        url = f"{self.binance_base}/api/v3/klines"
        frames: List[pd.DataFrame] = []
        current_start = start_ms

        while current_start <= end_ms:
            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": limit_per_call,
                "startTime": current_start,
                "endTime": min(current_start + limit_span - 1, end_ms),
            }

            attempt = 0
            backoff = sleep_sec
            while attempt < 5:
                try:
                    response = requests.get(url, params=params, timeout=10)
                    response.raise_for_status()
                    raw = response.json()
                    break
                except requests.RequestException as exc:
                    attempt += 1
                    print(
                        f"✗ Binance request error for {symbol} (attempt {attempt}/5): {exc}"
                    )
                    time.sleep(backoff)
                    backoff *= 2
            else:
                print("✗ Exhausted retries while fetching Binance history.")
                break

            if not raw:
                break

            frame = self._parse_crypto_klines(raw)
            if frame.empty:
                break

            frames.append(frame)

            last_close_ms = int(raw[-1][6])
            current_start = last_close_ms + 1

            if current_start > end_ms:
                break

            time.sleep(sleep_sec)

        if not frames:
            print("✗ No data returned during Binance pagination.")
            return pd.DataFrame()

        full_df = pd.concat(frames).sort_index()
        start_naive = start_utc.tz_localize(None)
        end_naive = end_utc.tz_localize(None)
        full_df = full_df.loc[(full_df.index >= start_naive) & (full_df.index <= end_naive)]
        full_df = full_df[~full_df.index.duplicated(keep="first")]

        print(
            f"✓ Fetched {len(full_df)} {interval} bars for {symbol} "
            f"between {full_df.index.min()} and {full_df.index.max()}"
        )

        if self._cache is not None and not full_df.empty:
            self._cache.save(full_df, **cache_key)

        return full_df

    def fetch_equity_ohlcv_range(
        self,
        symbol: str,
        start_dt: pd.Timestamp,
        end_dt: pd.Timestamp,
        interval: str = "5m",
    ) -> pd.DataFrame:
        """Fetch Yahoo Finance OHLCV over an explicit date range."""

        start_utc = self._ensure_utc(start_dt)
        end_utc = self._ensure_utc(end_dt)

        if start_utc >= end_utc:
            print("✗ Invalid date range supplied for equity fetch.")
            return pd.DataFrame()

        cache_key = {
            "source": "yfinance",
            "symbol": symbol,
            "interval": interval,
            "start": start_utc.isoformat(),
            "end": end_utc.isoformat(),
        }

        if self._cache is not None:
            cached = self._cache.load(**cache_key)
            if cached is not None and not cached.empty:
                print(
                    f"✓ Cache hit for {symbol} ({interval}) covering {len(cached)} bars"
                )
                return cached

        start_ts = start_utc.tz_localize(None)
        end_ts = end_utc.tz_localize(None)

        df = self.fetch_equity_ohlcv(
            symbol,
            period="7d",
            interval=interval,
            start=start_ts,
            end=end_ts,
        )

        if self._cache is not None and not df.empty:
            self._cache.save(df, **cache_key)

        return df

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
