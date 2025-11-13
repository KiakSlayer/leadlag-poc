"""
Correlation Analyzer for Cross-Asset Lead-Lag System
====================================================
Computes correlations and lead-lag relationships between asset pairs
"""

import pandas as pd
import numpy as np
from itertools import combinations, product
from typing import Dict, List, Tuple, Optional, Sequence
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import coint, adfuller
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
        if self.prices.empty or len(self.prices) < 2:
            return pd.DataFrame()

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

    def find_pairs(
        self,
        assets_a: List[str],
        assets_b: Optional[List[str]] = None,
        min_correlation: float = 0.3
    ) -> List[Tuple[str, str, float]]:
        """
        Find pairs of assets with significant correlations.

        Args:
            assets_a: List of asset names for the first leg
            assets_b: Optional list of asset names for the second leg. When
                omitted, all unique combinations within ``assets_a`` are
                evaluated.
            min_correlation: Minimum absolute correlation threshold

        Returns:
            List of (asset1, asset2, correlation) tuples sorted by absolute
            correlation strength.
        """
        corr_matrix = self.calculate_correlation_matrix()
        pairs: List[Tuple[str, str, float]] = []

        if assets_b is None:
            combos = combinations(assets_a, 2)
        else:
            combos = product(assets_a, assets_b)

        for asset1, asset2 in combos:
            if asset1 == asset2:
                continue
            if asset1 not in corr_matrix.columns or asset2 not in corr_matrix.columns:
                continue

            corr_value = corr_matrix.loc[asset1, asset2]

            if pd.isna(corr_value):
                continue

            if abs(corr_value) >= min_correlation:
                pairs.append((asset1, asset2, corr_value))

        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        return pairs

    def filter_cointegrated_pairs(
        self,
        pairs: Sequence[Tuple[str, str, float]],
        significance_level: float = 0.05,
        max_lag: Optional[int] = None,
        autolag: str = "aic",
        min_samples: int = 60,
    ) -> pd.DataFrame:
        """Filter candidate pairs using the Engle-Granger cointegration test.

        Args:
            pairs: Iterable of (asset1, asset2, correlation) tuples.
            significance_level: Maximum p-value threshold to accept a pair.
            max_lag: Optional maximum lag parameter forwarded to ``statsmodels``.
            autolag: Information criterion for lag selection.
            min_samples: Minimum number of overlapping observations required to
                evaluate cointegration.

        Returns:
            DataFrame with the surviving pairs sorted by p-value and
            correlation strength. The columns include ``asset1``, ``asset2``,
            ``correlation``, ``p_value`` and the Engle-Granger test statistic.
        """

        qualified_pairs: List[Dict[str, float]] = []

        for asset1, asset2, corr_value in pairs:
            if asset1 not in self.prices.columns or asset2 not in self.prices.columns:
                continue

            pair_prices = self.prices[[asset1, asset2]].dropna()

            if len(pair_prices) < min_samples:
                continue

            series1 = np.log(pair_prices[asset1].astype(float))
            series2 = np.log(pair_prices[asset2].astype(float))

            try:
                stat, p_value, crit_values = coint(
                    series1,
                    series2,
                    trend="c",
                    maxlag=max_lag,
                    autolag=autolag,
                )
            except Exception:
                continue

            if not np.isfinite(p_value):
                continue

            if p_value < significance_level:
                qualified_pairs.append(
                    {
                        "asset1": asset1,
                        "asset2": asset2,
                        "correlation": corr_value,
                        "p_value": float(p_value),
                        "test_stat": float(stat),
                        "crit_value_5pct": float(crit_values[1]) if len(crit_values) > 1 else np.nan,
                    }
                )

        if not qualified_pairs:
            return pd.DataFrame(
                columns=[
                    "asset1",
                    "asset2",
                    "correlation",
                    "p_value",
                    "test_stat",
                    "crit_value_5pct",
                ]
            )

        df_pairs = pd.DataFrame(qualified_pairs)
        return df_pairs.sort_values(["p_value", "correlation"], ascending=[True, False])

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
        return self.find_pairs(crypto_assets, index_assets, min_correlation)

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

        best_lag = 0
        best_corr = 0.0

        for lag in range(-max_lag, max_lag + 1):
            shifted_returns2 = returns2.shift(-lag)
            aligned = pd.concat([returns1, shifted_returns2], axis=1, join="inner").dropna()

            if len(aligned) < 10:
                continue

            try:
                corr, _ = pearsonr(aligned.iloc[:, 0], aligned.iloc[:, 1])
            except Exception:
                corr = np.nan

            if not np.isfinite(corr):
                continue

            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag

        return best_lag, best_corr

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

        # Handle empty DataFrame
        if df.empty or len(df) == 0:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=[
                'pair', 'leader', 'lagger', 'base_correlation',
                'lag_periods', 'max_correlation', 'relationship'
            ])

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

    def test_cointegration(
        self,
        asset1: str,
        asset2: str,
        significance_level: float = 0.05
    ) -> Dict[str, any]:
        """
        Perform Engle-Granger cointegration test on two assets

        Cointegration tests whether two non-stationary time series have a stable
        long-term relationship. If cointegrated, the linear combination (spread)
        is stationary, making mean-reversion strategies more reliable.

        Args:
            asset1: First asset name
            asset2: Second asset name
            significance_level: P-value threshold (default: 0.05)

        Returns:
            Dictionary with:
                - cointegrated: Boolean (True if p-value < significance_level)
                - p_value: P-value from Engle-Granger test
                - test_statistic: Test statistic value
                - critical_values: Dict of critical values at different levels
                - hedge_ratio: Estimated hedge ratio (beta) from OLS
        """
        # Get price series
        if asset1 not in self.prices.columns or asset2 not in self.prices.columns:
            return {
                'cointegrated': False,
                'p_value': 1.0,
                'test_statistic': 0.0,
                'critical_values': {},
                'hedge_ratio': 0.0,
                'error': f'Asset not found'
            }

        series1 = self.prices[asset1].dropna()
        series2 = self.prices[asset2].dropna()

        # Align series
        common_idx = series1.index.intersection(series2.index)
        series1 = series1.loc[common_idx]
        series2 = series2.loc[common_idx]

        if len(series1) < 30:
            return {
                'cointegrated': False,
                'p_value': 1.0,
                'test_statistic': 0.0,
                'critical_values': {},
                'hedge_ratio': 0.0,
                'error': 'Insufficient data (<30 points)'
            }

        try:
            # Perform Engle-Granger cointegration test
            # Returns: (test_statistic, p_value, critical_values)
            test_stat, p_value, crit_values = coint(series1, series2)

            # Calculate hedge ratio (OLS beta)
            # y = alpha + beta * x
            x = series2.values.reshape(-1, 1)
            y = series1.values

            # Simple OLS: beta = cov(x,y) / var(x)
            x_mean = x.mean()
            y_mean = y.mean()
            cov_xy = ((x.flatten() - x_mean) * (y - y_mean)).mean()
            var_x = ((x.flatten() - x_mean) ** 2).mean()

            hedge_ratio = cov_xy / var_x if var_x != 0 else 0.0

            # Format critical values dictionary
            critical_values_dict = {}
            if isinstance(crit_values, np.ndarray) and len(crit_values) >= 3:
                critical_values_dict = {
                    '1%': float(crit_values[0]),
                    '5%': float(crit_values[1]),
                    '10%': float(crit_values[2])
                }

            return {
                'cointegrated': p_value < significance_level,
                'p_value': float(p_value),
                'test_statistic': float(test_stat),
                'critical_values': critical_values_dict,
                'hedge_ratio': float(hedge_ratio)
            }

        except Exception as e:
            return {
                'cointegrated': False,
                'p_value': 1.0,
                'test_statistic': 0.0,
                'critical_values': {},
                'hedge_ratio': 0.0,
                'error': str(e)
            }

    def filter_cointegrated_pairs(
        self,
        pairs: List[Tuple[str, str, float]],
        significance_level: float = 0.05,
        verbose: bool = True
    ) -> List[Tuple[str, str, float, Dict]]:
        """
        Filter pairs by cointegration test

        This is the KEY FUNCTION for Kiak's deliverable!
        Only keeps pairs that pass the Engle-Granger cointegration test (p < 0.05)

        Args:
            pairs: List of (asset1, asset2, correlation) tuples
            significance_level: P-value threshold (default: 0.05)
            verbose: Print detailed results

        Returns:
            List of (asset1, asset2, correlation, coint_results) tuples
            Only includes cointegrated pairs
        """
        cointegrated_pairs = []

        if verbose:
            print(f"\n{'='*70}")
            print(f"🧪 COINTEGRATION TEST (Engle-Granger)")
            print(f"   Significance Level: {significance_level}")
            print(f"   Testing {len(pairs)} pairs...")
            print(f"{'='*70}\n")

        for asset1, asset2, correlation in pairs:
            coint_result = self.test_cointegration(asset1, asset2, significance_level)

            if verbose:
                status = "✓ PASS" if coint_result['cointegrated'] else "✗ FAIL"
                print(f"{status} | {asset1:15s} <-> {asset2:10s}")
                print(f"        P-value: {coint_result['p_value']:.4f} | Corr: {correlation:+.4f} | β: {coint_result.get('hedge_ratio', 0):.4f}")

                if 'error' in coint_result:
                    print(f"        Error: {coint_result['error']}")

                if coint_result['cointegrated']:
                    print(f"        ✓ Cointegrated (stable long-term relationship)")
                else:
                    print(f"        ✗ Not cointegrated (unstable relationship)")
                print()

            if coint_result['cointegrated']:
                cointegrated_pairs.append((asset1, asset2, correlation, coint_result))

        if verbose:
            print(f"{'='*70}")
            print(f"RESULTS: {len(cointegrated_pairs)}/{len(pairs)} pairs are cointegrated")
            print(f"{'='*70}\n")

        return cointegrated_pairs

    def compare_correlation_vs_cointegration(
        self,
        pairs: List[Tuple[str, str, float]]
    ) -> pd.DataFrame:
        """
        Compare correlation-based vs cointegration-based pair selection

        Args:
            pairs: List of (asset1, asset2, correlation) tuples

        Returns:
            DataFrame with comparison metrics for each pair
        """
        results = []

        for asset1, asset2, correlation in pairs:
            coint_result = self.test_cointegration(asset1, asset2)

            results.append({
                'pair': f"{asset1}-{asset2}",
                'correlation': correlation,
                'cointegrated': coint_result['cointegrated'],
                'p_value': coint_result['p_value'],
                'test_statistic': coint_result['test_statistic'],
                'hedge_ratio': coint_result.get('hedge_ratio', 0.0),
                'pass_coint': 'YES' if coint_result['cointegrated'] else 'NO'
            })

        df = pd.DataFrame(results)
        return df.sort_values('p_value')


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
