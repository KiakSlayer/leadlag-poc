"""
Cross-Asset Lead-Lag Z-Score Model
==================================
Implements lead-lag arbitrage strategy using Z-score signals
for crypto-index pairs
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ModelConfig:
    """Configuration for lead-lag model"""
    window: int = 60  # Rolling window for statistics
    z_entry: float = 2.0  # Z-score threshold for entry
    z_exit: float = 0.5  # Z-score threshold for exit
    lag_periods: int = 0  # Lead-lag offset (positive if leader leads)


class CrossAssetLeadLagModel:
    """
    Lead-lag Z-score model for cross-asset pairs

    Strategy:
    1. Calculate hedge ratio (beta) between leader and lagger
    2. Compute spread = r_leader - beta * r_lagger
    3. Calculate Z-score of spread
    4. Generate signals:
       - Z > z_entry → SHORT leader, LONG lagger
       - Z < -z_entry → LONG leader, SHORT lagger
       - |Z| < z_exit → FLAT (close positions)
    """

    def __init__(self, config: ModelConfig = ModelConfig()):
        """
        Initialize model with configuration

        Args:
            config: Model configuration parameters
        """
        self.config = config
        self.signals_history = []

    def calculate_beta(
        self,
        returns_leader: np.ndarray,
        returns_lagger: np.ndarray
    ) -> float:
        """
        Calculate OLS hedge ratio (beta)

        beta = cov(leader, lagger) / var(lagger)

        Args:
            returns_leader: Leader asset returns
            returns_lagger: Lagger asset returns

        Returns:
            Beta coefficient (hedge ratio)
        """
        # Remove NaN values pairwise
        mask = np.isfinite(returns_leader) & np.isfinite(returns_lagger)
        leader = returns_leader[mask]
        lagger = returns_lagger[mask]

        if len(leader) < 2:
            return np.nan

        # Calculate covariance and variance
        mean_leader = leader.mean()
        mean_lagger = lagger.mean()

        cov = ((leader - mean_leader) * (lagger - mean_lagger)).mean()
        var_lagger = ((lagger - mean_lagger) ** 2).mean()

        if not np.isfinite(var_lagger) or var_lagger == 0.0:
            return np.nan

        beta = cov / var_lagger
        return beta

    def calculate_spread(
        self,
        returns_leader: np.ndarray,
        returns_lagger: np.ndarray,
        beta: float
    ) -> np.ndarray:
        """
        Calculate spread series

        spread = r_leader - beta * r_lagger

        Args:
            returns_leader: Leader returns
            returns_lagger: Lagger returns
            beta: Hedge ratio

        Returns:
            Spread series
        """
        spread = returns_leader - beta * returns_lagger
        return spread

    def calculate_zscore(self, spread: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate Z-score of spread

        Z = (spread - mean(spread)) / std(spread)

        Args:
            spread: Spread series

        Returns:
            Tuple of (mean, std, zscore_current)
        """
        # Remove NaN values
        spread_clean = spread[np.isfinite(spread)]

        if len(spread_clean) < 2:
            return np.nan, np.nan, np.nan

        mean = spread_clean.mean()
        std = spread_clean.std(ddof=0)

        if not np.isfinite(std) or std == 0.0:
            return mean, std, np.nan

        # Z-score of most recent spread
        zscore = (spread_clean[-1] - mean) / std

        return mean, std, zscore

    def generate_signal(self, zscore: float, prev_signal: str = "FLAT") -> str:
        """
        Generate trading signal based on Z-score

        Args:
            zscore: Current Z-score value
            prev_signal: Previous signal state

        Returns:
            Signal string: "LONG_leader_SHORT_lagger", "SHORT_leader_LONG_lagger", "FLAT", "HOLD"
        """
        if not np.isfinite(zscore):
            return "HOLD"

        # Entry signals
        if zscore > self.config.z_entry:
            # Spread is too high → mean reversion expected
            # Short the leader, long the lagger
            return "SHORT_leader_LONG_lagger"

        elif zscore < -self.config.z_entry:
            # Spread is too low → mean reversion expected
            # Long the leader, short the lagger
            return "LONG_leader_SHORT_lagger"

        # Exit signal
        elif abs(zscore) < self.config.z_exit:
            # Spread has reverted to mean
            return "FLAT"

        # Hold current position
        else:
            return "HOLD"

    def apply_lead_lag_offset(
        self,
        returns_leader: pd.Series,
        returns_lagger: pd.Series,
        lag: int
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Apply lead-lag offset to align series

        Args:
            returns_leader: Leader returns series
            returns_lagger: Lagger returns series
            lag: Lead-lag periods (positive if leader leads)

        Returns:
            Tuple of aligned (leader, lagger) series
        """
        if lag == 0:
            return returns_leader, returns_lagger

        if lag > 0:
            # Leader leads: shift lagger forward
            leader_aligned = returns_leader.iloc[lag:]
            lagger_aligned = returns_lagger.iloc[:-lag]
        else:
            # Lagger leads: shift leader forward
            leader_aligned = returns_leader.iloc[:lag]
            lagger_aligned = returns_lagger.iloc[-lag:]

        # Ensure indices align
        common_idx = leader_aligned.index.intersection(lagger_aligned.index)
        return leader_aligned.loc[common_idx], lagger_aligned.loc[common_idx]

    def run_strategy(
        self,
        prices: pd.DataFrame,
        leader: str,
        lagger: str,
        lag: int = 0
    ) -> pd.DataFrame:
        """
        Run lead-lag strategy on price data

        Args:
            prices: DataFrame with asset prices (columns are assets)
            leader: Name of leader asset
            lagger: Name of lagger asset
            lag: Lead-lag offset periods

        Returns:
            DataFrame with signals and strategy metrics
        """
        # Calculate returns
        returns = np.log(prices / prices.shift(1))

        # Apply lead-lag offset if needed
        if lag != 0:
            r_leader, r_lagger = self.apply_lead_lag_offset(
                returns[leader], returns[lagger], lag
            )
        else:
            r_leader = returns[leader]
            r_lagger = returns[lagger]

        # Align on common index
        common_idx = r_leader.index.intersection(r_lagger.index)

        # Check if there's any common data
        if len(common_idx) == 0:
            print(f"⚠️  Warning: No common timestamps between {leader} and {lagger}")
            # Return empty DataFrame with expected structure
            empty_df = pd.DataFrame(columns=[
                'r_leader', 'r_lagger', 'beta', 'spread',
                'spread_mean', 'spread_std', 'zscore', 'signal', 'raw_signal'
            ])
            empty_df.index.name = 'timestamp'
            return empty_df

        r_leader = r_leader.loc[common_idx]
        r_lagger = r_lagger.loc[common_idx]

        # Check if we have enough data for the window
        if len(r_leader) < self.config.window:
            print(f"⚠️  Warning: Insufficient data for {leader}-{lagger}")
            print(f"   Need at least {self.config.window} points, got {len(r_leader)}")
            # Return empty DataFrame
            empty_df = pd.DataFrame(columns=[
                'r_leader', 'r_lagger', 'beta', 'spread',
                'spread_mean', 'spread_std', 'zscore', 'signal', 'raw_signal'
            ])
            empty_df.index.name = 'timestamp'
            return empty_df

        # Initialize results
        results = []
        prev_state = "FLAT"

        # Rolling window calculation
        for i in range(len(r_leader)):
            if i < self.config.window - 1:
                # Not enough data yet to form a full window
                results.append({
                    'timestamp': r_leader.index[i],
                    'r_leader': r_leader.iloc[i],
                    'r_lagger': r_lagger.iloc[i],
                    'beta': np.nan,
                    'spread': np.nan,
                    'spread_mean': np.nan,
                    'spread_std': np.nan,
                    'zscore': np.nan,
                    'signal': prev_state,
                    'raw_signal': "HOLD"
                })
                continue

            # Get window data
            start_idx = i - (self.config.window - 1)
            window_leader = r_leader.iloc[start_idx:i + 1].values
            window_lagger = r_lagger.iloc[start_idx:i + 1].values

            # Calculate beta
            beta = self.calculate_beta(window_leader, window_lagger)

            # Calculate spread
            spread = self.calculate_spread(window_leader, window_lagger, beta)

            # Calculate Z-score
            spread_mean, spread_std, zscore = self.calculate_zscore(spread)

            # Generate signal
            raw_signal = self.generate_signal(zscore, prev_state)
            if raw_signal == "HOLD":
                signal = prev_state
            else:
                signal = raw_signal

            prev_state = signal

            results.append({
                'timestamp': r_leader.index[i],
                'r_leader': r_leader.iloc[i],
                'r_lagger': r_lagger.iloc[i],
                'beta': beta,
                'spread': spread[-1] if np.isfinite(beta) else np.nan,
                'spread_mean': spread_mean,
                'spread_std': spread_std,
                'zscore': zscore,
                'signal': signal,
                'raw_signal': raw_signal
            })

        # Create DataFrame from results
        df_signals = pd.DataFrame(results)

        # Handle empty results
        if df_signals.empty or len(df_signals) == 0:
            # Return empty DataFrame with expected structure
            empty_df = pd.DataFrame(columns=[
                'r_leader', 'r_lagger', 'beta', 'spread',
                'spread_mean', 'spread_std', 'zscore', 'signal', 'raw_signal'
            ])
            empty_df.index.name = 'timestamp'
            return empty_df

        # Set timestamp as index
        df_signals.set_index('timestamp', inplace=True)

        return df_signals

    def run_multiple_pairs(
        self,
        prices: pd.DataFrame,
        pairs: List[Tuple[str, str, int]]
    ) -> Dict[str, pd.DataFrame]:
        """
        Run strategy on multiple pairs

        Args:
            prices: DataFrame with all asset prices
            pairs: List of (leader, lagger, lag) tuples

        Returns:
            Dictionary of {pair_name: signals_df}
        """
        results = {}

        for leader, lagger, lag in pairs:
            pair_name = f"{leader}-{lagger}"
            print(f"Processing {pair_name} (lag={lag})...")

            signals_df = self.run_strategy(prices, leader, lagger, lag)
            results[pair_name] = signals_df

        return results


if __name__ == "__main__":
    # Test the model
    from data_fetcher import DataFetcher
    from correlation_analyzer import CorrelationAnalyzer

    print("=" * 60)
    print("Testing Cross-Asset Lead-Lag Model")
    print("=" * 60)

    # Fetch data
    fetcher = DataFetcher()
    data = fetcher.fetch_all_assets(
        crypto_symbols=['BTCUSDT', 'ETHUSDT'],
        equity_symbols={'SP500': '^GSPC', 'NASDAQ': '^IXIC'},
        period="5d",
        interval="1m"
    )

    aligned_data = fetcher.align_timestamps(data, method="inner")
    prices = fetcher.get_close_prices(aligned_data)

    # Analyze correlations to find best pairs
    analyzer = CorrelationAnalyzer(prices)
    crypto_assets = ['BTCUSDT', 'ETHUSDT']
    index_assets = ['SP500', 'NASDAQ']

    best_pairs = analyzer.find_best_pairs(crypto_assets, index_assets, min_correlation=0.0)
    lead_lag_analysis = analyzer.analyze_all_pairs(best_pairs[:2], max_lag=3)

    print("\n" + "=" * 60)
    print("Lead-Lag Analysis Results:")
    print("=" * 60)
    print(lead_lag_analysis.to_string(index=False))

    # Run model on best pair
    if len(lead_lag_analysis) > 0:
        best_pair = lead_lag_analysis.iloc[0]
        leader = best_pair['leader']
        lagger = best_pair['lagger']
        lag = int(best_pair['lag_periods'])

        print(f"\n" + "=" * 60)
        print(f"Running Strategy: {leader} -> {lagger} (lag={lag})")
        print("=" * 60)

        config = ModelConfig(window=60, z_entry=2.0, z_exit=0.5)
        model = CrossAssetLeadLagModel(config)

        signals = model.run_strategy(prices, leader, lagger, lag)

        print("\nSignals Summary:")
        print(signals.tail(10))

        print("\nSignal Distribution:")
        print(signals['signal'].value_counts())
