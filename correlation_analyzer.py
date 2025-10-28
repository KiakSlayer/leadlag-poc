"""
Correlation Analyzer for Cross-Asset Lead-Lag System
====================================================
Computes correlations and lead-lag relationships between asset pairs
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')


class CorrelationAnalyzer:
    """Analyzes correlations and lead-lag relationships between assets"""

    def __init__(self, prices_df: pd.DataFrame):
        """
        Initialize with price data

        Args:
            prices_df: DataFrame with timestamps as index and assets as columns
        """
        self.prices = prices_df
        self.returns = self._calculate_returns()

    def _calculate_returns(self) -> pd.DataFrame:
        """Calculate log returns for all assets"""
        returns = np.log(self.prices / self.prices.shift(1))
        return returns.dropna()

    def calculate_correlation_matrix(
        self,
        window: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix between all assets

        Args:
            window: Rolling window size (None for full period)

        Returns:
            Correlation matrix DataFrame
        """
        if window:
            corr = self.returns.rolling(window).corr().iloc[-len(self.returns.columns):]
        else:
            corr = self.returns.corr()

        return corr

    def find_best_pairs(
        self,
        crypto_assets: List[str],
        index_assets: List[str],
        min_correlation: float = 0.3
    ) -> List[Tuple[str, str, float]]:
        """
        Find crypto-index pairs with significant correlations

        Args:
            crypto_assets: List of crypto asset names
            index_assets: List of index asset names
            min_correlation: Minimum absolute correlation threshold

        Returns:
            List of (crypto, index, correlation) tuples, sorted by |correlation|
        """
        corr_matrix = self.calculate_correlation_matrix()
        pairs = []

        for crypto in crypto_assets:
            if crypto not in corr_matrix.columns:
                continue
            for index in index_assets:
                if index not in corr_matrix.columns:
                    continue

                corr_value = corr_matrix.loc[crypto, index]

                if abs(corr_value) >= min_correlation:
                    pairs.append((crypto, index, corr_value))

        # Sort by absolute correlation (strongest first)
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)

        return pairs

    def detect_lead_lag(
        self,
        asset1: str,
        asset2: str,
        max_lag: int = 10
    ) -> Tuple[int, float]:
        """
        Detect lead-lag relationship between two assets using cross-correlation

        Args:
            asset1: First asset name
            asset2: Second asset name
            max_lag: Maximum lag to test (in periods)

        Returns:
            Tuple of (optimal_lag, max_correlation)
            - Positive lag: asset1 leads asset2
            - Negative lag: asset2 leads asset1
        """
        returns1 = self.returns[asset1].dropna()
        returns2 = self.returns[asset2].dropna()

        # Align the series
        common_idx = returns1.index.intersection(returns2.index)
        returns1 = returns1.loc[common_idx]
        returns2 = returns2.loc[common_idx]

        correlations = []
        lags = range(-max_lag, max_lag + 1)

        for lag in lags:
            if lag < 0:
                # asset2 leads asset1
                r1 = returns1.iloc[-lag:]
                r2 = returns2.iloc[:lag]
            elif lag > 0:
                # asset1 leads asset2
                r1 = returns1.iloc[:-lag]
                r2 = returns2.iloc[lag:]
            else:
                r1 = returns1
                r2 = returns2

            # Align indices
            common = r1.index.intersection(r2.index)
            if len(common) < 10:  # Need minimum data points
                correlations.append(0.0)
                continue

            r1_aligned = r1.loc[common]
            r2_aligned = r2.loc[common]

            try:
                corr, _ = pearsonr(r1_aligned, r2_aligned)
                correlations.append(corr if np.isfinite(corr) else 0.0)
            except:
                correlations.append(0.0)

        # Find lag with maximum absolute correlation
        abs_corrs = [abs(c) for c in correlations]
        max_idx = np.argmax(abs_corrs)
        optimal_lag = lags[max_idx]
        max_correlation = correlations[max_idx]

        return optimal_lag, max_correlation

    def analyze_all_pairs(
        self,
        pairs: List[Tuple[str, str, float]],
        max_lag: int = 10
    ) -> pd.DataFrame:
        """
        Analyze lead-lag relationships for all pairs

        Args:
            pairs: List of (asset1, asset2, correlation) tuples
            max_lag: Maximum lag to test

        Returns:
            DataFrame with analysis results
        """
        results = []

        for asset1, asset2, base_corr in pairs:
            lag, max_corr = self.detect_lead_lag(asset1, asset2, max_lag)

            lead_lag_str = ""
            if lag > 0:
                lead_lag_str = f"{asset1} leads {asset2} by {lag} periods"
                leader = asset1
                lagger = asset2
            elif lag < 0:
                lead_lag_str = f"{asset2} leads {asset1} by {abs(lag)} periods"
                leader = asset2
                lagger = asset1
            else:
                lead_lag_str = "Simultaneous"
                leader = asset1
                lagger = asset2

            results.append({
                'pair': f"{asset1}-{asset2}",
                'leader': leader,
                'lagger': lagger,
                'base_correlation': base_corr,
                'lag_periods': lag,
                'max_correlation': max_corr,
                'relationship': lead_lag_str
            })

        df = pd.DataFrame(results)
        return df.sort_values('max_correlation', ascending=False, key=abs)

    def rolling_correlation(
        self,
        asset1: str,
        asset2: str,
        window: int = 60
    ) -> pd.Series:
        """
        Calculate rolling correlation between two assets

        Args:
            asset1: First asset name
            asset2: Second asset name
            window: Rolling window size

        Returns:
            Series of rolling correlations
        """
        returns1 = self.returns[asset1]
        returns2 = self.returns[asset2]

        rolling_corr = returns1.rolling(window).corr(returns2)
        return rolling_corr

    def get_correlation_stability(
        self,
        asset1: str,
        asset2: str,
        window: int = 60
    ) -> Dict[str, float]:
        """
        Measure stability of correlation over time

        Args:
            asset1: First asset name
            asset2: Second asset name
            window: Rolling window size

        Returns:
            Dictionary with stability metrics
        """
        rolling_corr = self.rolling_correlation(asset1, asset2, window)
        rolling_corr = rolling_corr.dropna()

        if len(rolling_corr) < 2:
            return {
                'mean_correlation': 0.0,
                'std_correlation': 0.0,
                'min_correlation': 0.0,
                'max_correlation': 0.0,
                'stability_score': 0.0
            }

        mean_corr = rolling_corr.mean()
        std_corr = rolling_corr.std()
        min_corr = rolling_corr.min()
        max_corr = rolling_corr.max()

        # Stability score: higher mean, lower std is better
        stability_score = abs(mean_corr) / (std_corr + 0.01)  # Avoid division by zero

        return {
            'mean_correlation': mean_corr,
            'std_correlation': std_corr,
            'min_correlation': min_corr,
            'max_correlation': max_corr,
            'stability_score': stability_score
        }


if __name__ == "__main__":
    # Test the correlation analyzer
    from data_fetcher import DataFetcher

    print("=" * 60)
    print("Testing Correlation Analyzer")
    print("=" * 60)

    # Fetch data
    fetcher = DataFetcher()
    data = fetcher.fetch_all_assets(
        crypto_symbols=['BTCUSDT', 'ETHUSDT', 'SOLUSDT'],
        equity_symbols={'SP500': '^GSPC', 'NASDAQ': '^IXIC'},
        period="5d",
        interval="1m"
    )

    aligned_data = fetcher.align_timestamps(data, method="inner")
    prices = fetcher.get_close_prices(aligned_data)

    # Analyze correlations
    analyzer = CorrelationAnalyzer(prices)

    print("\n" + "=" * 60)
    print("Correlation Matrix:")
    print("=" * 60)
    corr_matrix = analyzer.calculate_correlation_matrix()
    print(corr_matrix)

    print("\n" + "=" * 60)
    print("Best Crypto-Index Pairs:")
    print("=" * 60)
    crypto_assets = [col for col in prices.columns if 'USDT' in col]
    index_assets = [col for col in prices.columns if 'USDT' not in col]

    best_pairs = analyzer.find_best_pairs(crypto_assets, index_assets, min_correlation=0.0)

    for crypto, index, corr in best_pairs:
        print(f"{crypto:15s} <-> {index:10s}: {corr:+.4f}")

    print("\n" + "=" * 60)
    print("Lead-Lag Analysis:")
    print("=" * 60)
    analysis = analyzer.analyze_all_pairs(best_pairs[:5], max_lag=5)
    print(analysis.to_string(index=False))
