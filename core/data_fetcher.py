"""Data acquisition utilities for the cross-asset lead-lag system."""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import requests
import yfinance as yf

TimestampLike = Optional[Union[str, int, float, datetime, pd.Timestamp]]


class DataFetcher:
    """Unified OHLCV loader with caching and multiple data sources."""

    def __init__(
        self,
        cache_dir: Union[str, Path] = "data_cache",
        cache_ttl_hours: int = 12,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = timedelta(hours=cache_ttl_hours)

        # Symbol mapping: User-friendly names to Yahoo Finance tickers
        self.crypto_symbol_map = {
            "BTCUSDT": "BTC-USD",
            "ETHUSDT": "ETH-USD",
            "SOLUSDT": "SOL-USD",
            "BNBUSDT": "BNB-USD",
            "ADAUSDT": "ADA-USD",
            "XRPUSDT": "XRP-USD",
            "DOGEUSDT": "DOGE-USD",
            "DOTUSDT": "DOT-USD",
            "MATICUSDT": "MATIC-USD",
            "AVAXUSDT": "AVAX-USD",
        }

    # ------------------------------------------------------------------
    # Symbol helpers & general utilities
    # ------------------------------------------------------------------
    def _convert_symbol_to_yahoo(self, symbol: str) -> str:
        """Convert Binance-style symbols to Yahoo tickers when possible."""

        if "-" in symbol or "^" in symbol:
            return symbol

        if symbol in self.crypto_symbol_map:
            return self.crypto_symbol_map[symbol]

        if symbol.endswith("USDT"):
            base = symbol[:-4]
            return f"{base}-USD"

        return symbol

    def _interval_to_minutes(self, interval: str) -> float:
        mapping = {
            "1m": 1,
            "2m": 2,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "45m": 45,
            "1h": 60,
            "90m": 90,
            "2h": 120,
            "4h": 240,
            "6h": 360,
            "12h": 720,
            "1d": 1440,
        }
        return mapping.get(interval, 1)

    def _interval_to_milliseconds(self, interval: str) -> int:
        return int(self._interval_to_minutes(interval) * 60 * 1000)

    def _calculate_period_from_limit(self, interval: str, limit: int) -> str:
        minutes_per_day = 1440 if self._interval_to_minutes(interval) <= 60 else 390
        total_days = (self._interval_to_minutes(interval) * limit) / minutes_per_day

        if total_days <= 1:
            return "1d"
        if total_days <= 5:
            return "5d"
        if total_days <= 7:
            return "7d"
        if total_days <= 30:
            return "1mo"
        if total_days <= 90:
            return "3mo"
        if total_days <= 180:
            return "6mo"
        if total_days <= 365:
            return "1y"
        if total_days <= 730:
            return "2y"
        if total_days <= 1825:
            return "5y"
        return "max"

    def _parse_timestamp(self, value: TimestampLike) -> Optional[pd.Timestamp]:
        if value is None:
            return None
        try:
            if isinstance(value, (int, float)):
                return pd.to_datetime(int(value), unit="ms", utc=True).tz_convert(None)
            return pd.to_datetime(value, utc=True).tz_convert(None)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Cache utilities
    # ------------------------------------------------------------------
    def _cache_key(
        self,
        symbol: str,
        interval: str,
        source: str,
        start: Optional[pd.Timestamp],
        end: Optional[pd.Timestamp],
    ) -> Path:
        safe_symbol = symbol.replace("/", "_").replace(":", "_")
        start_key = start.strftime("%Y%m%d%H%M%S") if start is not None else "none"
        end_key = end.strftime("%Y%m%d%H%M%S") if end is not None else "none"
        filename = f"{safe_symbol}_{interval}_{source}_{start_key}_{end_key}.parquet"
        return self.cache_dir / filename

    def _load_from_cache(
        self,
        cache_path: Path,
    ) -> Optional[pd.DataFrame]:
        if not cache_path.exists():
            return None

        modified = datetime.utcfromtimestamp(cache_path.stat().st_mtime)
        if datetime.utcnow() - modified > self.cache_ttl:
            return None

        try:
            cached = pd.read_parquet(cache_path)
        except Exception as exc:  # pragma: no cover - cache read failure is non-critical
            print(f"  ⚠️  Failed to read cache {cache_path.name}: {exc}")
            return None

        if cached.empty:
            return None

        cached.index = pd.to_datetime(cached.index)
        return cached

    def _store_in_cache(self, cache_path: Path, df: pd.DataFrame) -> None:
        if df.empty:
            return
        try:
            df.to_parquet(cache_path)
        except Exception as exc:  # pragma: no cover - cache persistence is best-effort
            print(f"  ⚠️  Failed to store cache {cache_path.name}: {exc}")

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------
    def _finalise_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()

        work = df.copy()
        if "timestamp" not in work.columns or not pd.api.types.is_datetime64_any_dtype(
            work.get("timestamp")
        ):
            if "timestamp" in work.columns:
                work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True).tz_convert(None)
            elif work.index.name and pd.api.types.is_datetime64_any_dtype(work.index):
                work = work.reset_index().rename(columns={work.index.name: "timestamp"})
            else:
                first_col = work.columns[0]
                work = work.rename(columns={first_col: "timestamp"})
                work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True).tz_convert(None)

        if "timestamp" not in work.columns:
            raise ValueError("Failed to determine timestamp column for OHLCV data")

        work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True).tz_convert(None)
        work = work.set_index("timestamp").sort_index()

        essential_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in work.columns]
        return work[essential_cols]

    # ------------------------------------------------------------------
    # Data source implementations
    # ------------------------------------------------------------------
    def _fetch_yahoo_history(
        self,
        symbol: str,
        interval: str,
        limit: int,
        start: Optional[pd.Timestamp],
        end: Optional[pd.Timestamp],
    ) -> pd.DataFrame:
        ticker = yf.Ticker(symbol)
        period = None
        if start is None and end is None:
            period = self._calculate_period_from_limit(interval, limit)
            if period == "max":
                minutes = self._interval_to_minutes(interval)
                if minutes <= 0:
                    minutes = 1
                total_minutes = minutes * limit
                start = datetime.utcnow() - timedelta(minutes=total_minutes)
                start = pd.Timestamp(start, tz="UTC").tz_convert(None)
                period = None

        if period:
            df = ticker.history(period=period, interval=interval)
        else:
            df = ticker.history(start=start, end=end, interval=interval)

        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        return self._finalise_dataframe(df)

    def _fetch_binance_history(
        self,
        symbol: str,
        interval: str,
        start: Optional[pd.Timestamp],
        end: Optional[pd.Timestamp],
    ) -> pd.DataFrame:
        base_url = "https://api.binance.com/api/v3/klines"
        start_ms = int(start.timestamp() * 1000) if start is not None else None
        end_ms = int(end.timestamp() * 1000) if end is not None else None
        interval_ms = self._interval_to_milliseconds(interval)

        if end_ms is None:
            end_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
        if start_ms is None:
            # default to the maximum single request window
            start_ms = end_ms - (interval_ms * 1000)

        params = {"symbol": symbol.upper(), "interval": interval, "limit": 1000}
        frames: List[pd.DataFrame] = []
        current_start = start_ms

        while True:
            params["startTime"] = current_start
            if end_ms is not None:
                params["endTime"] = end_ms

            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            payload = response.json()
            if not payload:
                break

            batch = pd.DataFrame(
                payload,
                columns=[
                    "open_time",
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
            batch["timestamp"] = pd.to_datetime(batch["open_time"], unit="ms", utc=True).tz_convert(None)
            for col in ["open", "high", "low", "close", "volume"]:
                batch[col] = pd.to_numeric(batch[col], errors="coerce")
            frames.append(batch[["timestamp", "open", "high", "low", "close", "volume"]])

            last_open = int(payload[-1][0])
            next_start = last_open + interval_ms
            if end_ms is not None and next_start >= end_ms:
                break
            if len(payload) < 1000:
                break

            current_start = next_start
            time.sleep(0.25)  # polite rate limiting

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        return self._finalise_dataframe(combined)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fetch_crypto_ohlcv(
        self,
        symbol: str,
        interval: str = "1m",
        limit: int = 1000,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        start: TimestampLike = None,
        end: TimestampLike = None,
        source: str = "yahoo",
    ) -> pd.DataFrame:
        """Fetch crypto OHLCV data from Yahoo Finance or Binance."""

        start_ts = self._parse_timestamp(start) or self._parse_timestamp(start_time)
        end_ts = self._parse_timestamp(end) or self._parse_timestamp(end_time)

        cache_path = self._cache_key(symbol, interval, source, start_ts, end_ts)
        cached = self._load_from_cache(cache_path)
        if cached is not None:
            print(f"  ✓ Loaded {symbol} {interval} data from cache")
            return cached

        print(f"  Fetching {symbol} with interval={interval} from {source}...")

        if source.lower() == "binance":
            df = self._fetch_binance_history(symbol, interval, start_ts, end_ts)
        else:
            yahoo_symbol = self._convert_symbol_to_yahoo(symbol)
            df = self._fetch_yahoo_history(yahoo_symbol, interval, limit, start_ts, end_ts)

        if df.empty:
            print(f"  ✗ No data returned for {symbol}")
            return df

        self._store_in_cache(cache_path, df)
        print(f"  ✓ Fetched {len(df)} {interval} bars for {symbol}")
        return df

    def fetch_equity_ohlcv(
        self,
        symbol: str,
        period: str = "7d",
        interval: str = "1m",
        start: TimestampLike = None,
        end: TimestampLike = None,
    ) -> pd.DataFrame:
        """Fetch equity/index data from Yahoo Finance with caching."""

        start_ts = self._parse_timestamp(start)
        end_ts = self._parse_timestamp(end)

        cache_path = self._cache_key(symbol, interval, "yahoo", start_ts, end_ts)
        cached = self._load_from_cache(cache_path)
        if cached is not None:
            print(f"  ✓ Loaded {symbol} {interval} data from cache")
            return cached

        print(f"  Fetching {symbol} equity data interval={interval}...")

        if start_ts or end_ts:
            yahoo_df = self._fetch_yahoo_history(symbol, interval, limit=0, start=start_ts, end=end_ts)
        else:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]
            yahoo_df = self._finalise_dataframe(df)

        if yahoo_df.empty:
            print(f"  ✗ No data returned for {symbol}")
            return yahoo_df

        self._store_in_cache(cache_path, yahoo_df)
        print(f"  ✓ Fetched {len(yahoo_df)} {interval} bars for {symbol}")
        return yahoo_df

    def fetch_set50_ohlcv(
        self,
        period: str = "7d",
        interval: str = "1m",
        start: TimestampLike = None,
        end: TimestampLike = None,
    ) -> pd.DataFrame:
        """Fetch SET50 index data (Thailand) via Yahoo Finance."""

        return self.fetch_equity_ohlcv(
            symbol="^SET.BK",
            period=period,
            interval=interval,
            start=start,
            end=end,
        )

    def fetch_all_assets(
        self,
        crypto_symbols: List[str] = None,
        equity_symbols: Dict[str, str] = None,
        period: str = "7d",
        interval: str = "1m",
        start: TimestampLike = None,
        end: TimestampLike = None,
        crypto_source: str = "yahoo",
    ) -> Dict[str, pd.DataFrame]:
        """Fetch crypto and equity data in one call."""

        crypto_symbols = crypto_symbols or ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        equity_symbols = equity_symbols or {
            "SP500": "^GSPC",
            "NASDAQ": "^IXIC",
            "SET50": "^SET.BK",
        }

        all_data: Dict[str, pd.DataFrame] = {}

        print("\n📊 Fetching Crypto Data...")
        for symbol in crypto_symbols:
            df = self.fetch_crypto_ohlcv(
                symbol,
                interval=interval,
                limit=1000,
                start=start,
                end=end,
                source=crypto_source,
            )
            if not df.empty:
                all_data[symbol] = df
            time.sleep(0.2)

        print("\n📈 Fetching Equity Index Data...")
        for name, ticker in equity_symbols.items():
            df = self.fetch_equity_ohlcv(
                ticker,
                period=period,
                interval=interval,
                start=start,
                end=end,
            )
            if not df.empty:
                all_data[name] = df
            time.sleep(0.2)

        print(f"\n✓ Successfully fetched data for {len(all_data)} assets")
        return all_data

    # ------------------------------------------------------------------
    # Alignment helpers remain unchanged
    # ------------------------------------------------------------------
    def align_timestamps(
        self,
        data_dict: Dict[str, pd.DataFrame],
        method: str = "inner",
        tolerance: str = "1min",
    ) -> Dict[str, pd.DataFrame]:
        if not data_dict:
            return {}

        normalized_data = {}
        for name, df in data_dict.items():
            df_copy = df.copy()

            if df_copy.index.tz is not None:
                df_copy.index = df_copy.index.tz_convert("UTC").tz_localize(None)

            df_copy.index = df_copy.index.round("1min")
            df_copy = df_copy[~df_copy.index.duplicated(keep="first")]

            normalized_data[name] = df_copy

        all_indices = [df.index for df in normalized_data.values()]

        if method == "inner":
            common_index = all_indices[0]
            for idx in all_indices[1:]:
                common_index = common_index.intersection(idx)

            if len(common_index) == 0:
                print("⚠️  Warning: No exact timestamp overlap found.")
                print("   Switching to 'outer' join with forward fill...")
                print("   Asset date ranges:")
                for name, df in normalized_data.items():
                    print(
                        f"     {name:15s}: {df.index.min()} to {df.index.max()} ({len(df)} points)"
                    )

                common_index = all_indices[0]
                for idx in all_indices[1:]:
                    common_index = common_index.union(idx)
                common_index = common_index.sort_values()
        else:
            common_index = all_indices[0]
            for idx in all_indices[1:]:
                common_index = common_index.union(idx)
            common_index = common_index.sort_values()

        aligned_data = {}
        for name, df in normalized_data.items():
            aligned_df = df.reindex(common_index, method="ffill", limit=5)

            min_non_na = int(np.ceil(len(aligned_df.columns) * 0.5)) if len(aligned_df.columns) else 0
            if min_non_na > 0:
                aligned_df = aligned_df.dropna(thresh=min_non_na)

            aligned_data[name] = aligned_df

        if len(aligned_data) > 1:
            valid_indices = None
            for df in aligned_data.values():
                valid_idx = df.dropna().index
                if valid_indices is None:
                    valid_indices = valid_idx
                else:
                    valid_indices = valid_indices.intersection(valid_idx)

            if valid_indices is not None and len(valid_indices) > 0:
                aligned_data = {name: df.loc[valid_indices] for name, df in aligned_data.items()}
                common_index = valid_indices

        print(f"✓ Aligned data to {len(common_index)} common timestamps")

        if len(common_index) < 10:
            print(f"⚠️  Warning: Only {len(common_index)} common timestamps found!")
            print("   Consider using longer time period or different interval")

        return aligned_data

    def get_close_prices(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        close_prices = {}
        for name, df in data_dict.items():
            if "close" in df.columns:
                close_prices[name] = df["close"]

        return pd.DataFrame(close_prices)


if __name__ == "__main__":
    fetcher = DataFetcher()
    print("=" * 60)
    print("Testing Data Fetcher")
    print("=" * 60)

    data = fetcher.fetch_all_assets(
        crypto_symbols=["BTCUSDT", "ETHUSDT"],
        equity_symbols={"SP500": "^GSPC"},
        period="1mo",
        interval="1h",
    )

    aligned = fetcher.align_timestamps(data, method="inner")
    prices = fetcher.get_close_prices(aligned)
    print(prices.head())
