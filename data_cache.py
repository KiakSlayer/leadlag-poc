"""Simple Parquet-backed caching layer for historical market data."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Optional

import pandas as pd


class DataCache:
    """Persist pandas DataFrames keyed by a deterministic hash."""

    def __init__(self, root: str = "cache") -> None:
        Path(root).mkdir(parents=True, exist_ok=True)
        self.root = root

    def _key(self, **kwargs: Any) -> str:
        """Generate a filesystem path for the given keyword arguments."""
        key_repr = repr(sorted(kwargs.items()))
        digest = hashlib.md5(key_repr.encode(), usedforsecurity=False).hexdigest()
        return os.path.join(self.root, f"{digest}.parquet")

    def load(self, **kwargs: Any) -> Optional[pd.DataFrame]:
        """Return a cached dataframe for the provided key, if it exists."""
        path = self._key(**kwargs)
        if os.path.exists(path):
            return pd.read_parquet(path)
        return None

    def save(self, df: pd.DataFrame, **kwargs: Any) -> str:
        """Persist a dataframe under the derived cache key and return the path."""
        path = self._key(**kwargs)
        df.to_parquet(path)
        return path
