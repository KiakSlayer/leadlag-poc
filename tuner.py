#!/usr/bin/env python3
"""
Parameter Tuning System for Cross-Asset Lead-Lag Strategy
==========================================================

Systematically tests hyperparameter combinations and records performance metrics.

Author: Quantitative Development Team
Date: 2025
"""

import argparse
import sys
from datetime import datetime
from itertools import product
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from tqdm import tqdm

# Import existing modules
from data_fetcher import DataFetcher
from correlation_analyzer import CorrelationAnalyzer
from crossasset_leadlag_model import CrossAssetLeadLagModel, ModelConfig
from backtester import Backtester, BacktestConfig


class ParameterTuner:
    """
    Hyperparameter tuning system for lead-lag strategy
    
    Tests combinations of:
    - interval: Time interval for data
    - window: Rolling window size
    - z_entry: Entry Z-score threshold
    - z_exit: Exit Z-score threshold
    """
    
    def __init__(
        self,
        crypto_symbols=['BTCUSDT', 'ETHUSDT', 'SOLUSDT'],
        equity_symbols={'SP500': '^GSPC', 'NASDAQ': '^IXIC', 'SET50': '^SET.BK'},
        period='7d',
        initial_capital=100000.0,
        transaction_cost=0.001,
        min_correlation=0.05,
        max_lag=5
    ):
        """
        Initialize parameter tuner
        
        Args:
            crypto_symbols: List of crypto symbols
            equity_symbols: Dict of equity symbols
            period: Data period
            initial_capital: Starting capital for backtest
            transaction_cost: Transaction cost per trade
            min_correlation: Minimum correlation for pair selection
            max_lag: Maximum lag to test
        """
        self.crypto_symbols = crypto_symbols
        self.equity_symbols = equity_symbols
        self.period = period
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.min_correlation = min_correlation
        self.max_lag = max_lag
        self.results = []
        
    def fetch_and_prepare_data(self, interval: str) -> tuple:
        """
        Fetch and prepare data for given interval
        
        Args:
            interval: Data interval (e.g., '5m', '15m')
            
        Returns:
            Tuple of (prices_df, pairs_analysis)
        """
        print(f"\n📊 Fetching data with interval={interval}...")
        
        fetcher = DataFetcher()
        data = fetcher.fetch_all_assets(
            crypto_symbols=self.crypto_symbols,
            equity_symbols=self.equity_symbols,
            period=self.period,
            interval=interval
        )
        
        if len(data) == 0:
            return None, None
            
        # Align timestamps
        aligned_data = fetcher.align_timestamps(data, method="inner")
        prices = fetcher.get_close_prices(aligned_data)
        
        if len(prices) < 100:
            print(f"⚠️  Insufficient data for {interval}: {len(prices)} points")
            return None, None
            
        # Analyze correlations
        analyzer = CorrelationAnalyzer(prices)
        
        # Categorize assets
        crypto_assets = [col for col in prices.columns if 'USDT' in col or 'BTC' in col]
        index_assets = [col for col in prices.columns if col not in crypto_assets]
        
        # Find best pairs
        all_pairs = []
        
        # Crypto-Crypto pairs
        if len(crypto_assets) >= 2:
            crypto_pairs = analyzer.find_pairs(
                crypto_assets, min_correlation=self.min_correlation
            )
            for asset1, asset2, corr in crypto_pairs[:3]:
                all_pairs.append(('Crypto-Crypto', asset1, asset2, corr))
        
        # Crypto-Index pairs
        if crypto_assets and index_assets:
            crypto_index_pairs = analyzer.find_best_pairs(
                crypto_assets, index_assets, min_correlation=self.min_correlation
            )
            for asset1, asset2, corr in crypto_index_pairs[:3]:
                all_pairs.append(('Crypto-Index', asset1, asset2, corr))
        
        # Index-Index pairs
        if len(index_assets) >= 2:
            index_pairs = analyzer.find_pairs(
                index_assets, min_correlation=self.min_correlation
            )
            for asset1, asset2, corr in index_pairs[:3]:
                all_pairs.append(('Index-Index', asset1, asset2, corr))
        
        if not all_pairs:
            print(f"⚠️  No valid pairs found for {interval}")
            return None, None
            
        # Analyze lead-lag for all pairs
        pairs_with_leadlag = []
        for category, asset1, asset2, base_corr in all_pairs:
            lag, max_corr = analyzer.detect_lead_lag(asset1, asset2, max_lag=self.max_lag)
            
            if lag > 0:
                leader, lagger = asset1, asset2
            elif lag < 0:
                leader, lagger = asset2, asset1
                lag = abs(lag)
            else:
                leader, lagger = asset1, asset2
                
            pairs_with_leadlag.append({
                'category': category,
                'leader': leader,
                'lagger': lagger,
                'lag': lag,
                'base_corr': base_corr,
                'max_corr': max_corr
            })
        
        print(f"✓ Found {len(pairs_with_leadlag)} valid pairs with {len(prices)} data points")
        
        return prices, pairs_with_leadlag
    
    def test_configuration(
        self,
        prices: pd.DataFrame,
        pair_info: dict,
        window: int,
        z_entry: float,
        z_exit: float
    ) -> dict:
        """
        Test a single parameter configuration
        
        Args:
            prices: Price DataFrame
            pair_info: Dict with pair information
            window: Rolling window size
            z_entry: Entry Z-score threshold
            z_exit: Exit Z-score threshold
            
        Returns:
            Dict with performance metrics
        """
        # Create model config
        config = ModelConfig(
            window=window,
            z_entry=z_entry,
            z_exit=z_exit
        )
        
        model = CrossAssetLeadLagModel(config)
        
        # Run strategy
        signals = model.run_strategy(
            prices,
            pair_info['leader'],
            pair_info['lagger'],
            pair_info['lag']
        )
        
        # Check if signals were generated
        if signals.empty or len(signals) < window:
            return None
            
        # Run backtest
        bt_config = BacktestConfig(
            initial_capital=self.initial_capital,
            transaction_cost=self.transaction_cost,
            position_size=1.0
        )
        backtester = Backtester(bt_config)
        
        try:
            results = backtester.run_backtest(
                signals, prices,
                pair_info['leader'],
                pair_info['lagger']
            )
            
            metrics = results['metrics']
            
            return {
                'sharpe_ratio': metrics.get('sharpe_ratio', 0.0),
                'max_drawdown': metrics.get('max_drawdown', 0.0),
                'max_drawdown_pct': metrics.get('max_drawdown_pct', 0.0),
                'total_return_pct': metrics.get('total_return_pct', 0.0),
                'num_trades': metrics.get('num_trades', 0),
                'win_rate': metrics.get('win_rate', 0.0),
                'sortino_ratio': metrics.get('sortino_ratio', 0.0),
                'final_capital': metrics.get('final_capital', self.initial_capital),
                'volatility_annual': metrics.get('volatility_annual', 0.0)
            }
            
        except Exception as e:
            print(f"    ✗ Error in backtest: {e}")
            return None
    
    def run_parameter_sweep(
        self,
        intervals=['5m', '15m'],
        windows=[30, 60, 90],
        z_entries=[1.5, 2.0, 2.5],
        z_exits=[0.3, 0.5],
        max_pairs_per_interval=3
    ):
        """
        Run complete parameter sweep
        
        Args:
            intervals: List of data intervals to test
            windows: List of window sizes to test
            z_entries: List of entry thresholds to test
            z_exits: List of exit thresholds to test
            max_pairs_per_interval: Maximum pairs to test per interval
        """
        print("=" * 80)
        print("PARAMETER TUNING - CROSS-ASSET LEAD-LAG STRATEGY")
        print("=" * 80)
        print(f"\nConfiguration Space:")
        print(f"  Intervals: {intervals}")
        print(f"  Windows: {windows}")
        print(f"  Z-Entry: {z_entries}")
        print(f"  Z-Exit: {z_exits}")
        
        total_configs = len(intervals) * len(windows) * len(z_entries) * len(z_exits)
        print(f"\nTotal base configurations to test: {total_configs}")
        
        # Store data by interval to avoid refetching
        interval_data = {}
        
        for interval in intervals:
            prices, pairs = self.fetch_and_prepare_data(interval)
            
            if prices is None or pairs is None:
                continue
                
            interval_data[interval] = {
                'prices': prices,
                'pairs': pairs[:max_pairs_per_interval]
            }
        
        # Generate all parameter combinations
        param_combinations = list(product(
            intervals, windows, z_entries, z_exits
        ))
        
        print(f"\nStarting parameter sweep with {len(param_combinations)} configurations...")
        print(f"Estimated pairs to test: {sum(len(interval_data[i]['pairs']) for i in interval_data) * len(windows) * len(z_entries) * len(z_exits)}")
        print("=" * 80)
        
        # Progress bar
        with tqdm(total=len(param_combinations), desc="Testing configs") as pbar:
            for interval, window, z_entry, z_exit in param_combinations:
                
                if interval not in interval_data:
                    pbar.update(1)
                    continue
                
                data = interval_data[interval]
                prices = data['prices']
                pairs = data['pairs']
                
                # Test each pair with this configuration
                for pair_info in pairs:
                    
                    # Check if we have enough data for this window
                    if len(prices) < window + 10:
                        continue
                    
                    metrics = self.test_configuration(
                        prices, pair_info, window, z_entry, z_exit
                    )
                    
                    if metrics is None:
                        continue
                    
                    # Store result
                    result = {
                        'interval': interval,
                        'window': window,
                        'z_entry': z_entry,
                        'z_exit': z_exit,
                        'pair_category': pair_info['category'],
                        'leader': pair_info['leader'],
                        'lagger': pair_info['lagger'],
                        'lag': pair_info['lag'],
                        'base_correlation': pair_info['base_corr'],
                        'max_correlation': pair_info['max_corr'],
                        **metrics
                    }
                    
                    self.results.append(result)
                
                pbar.update(1)
        
        print(f"\n✓ Parameter sweep complete!")
        print(f"  Total valid results: {len(self.results)}")
    
    def save_results(self, filepath='tuning_results.csv'):
        """
        Save results to CSV
        
        Args:
            filepath: Path to save CSV file
        """
        if len(self.results) == 0:
            print("⚠️  No results to save")
            return
        
        df = pd.DataFrame(self.results)
        df.to_csv(filepath, index=False)
        print(f"\n✓ Results saved to {filepath}")
        print(f"  Shape: {df.shape}")
        print(f"\nColumn summary:")
        print(df.dtypes)
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Get results as DataFrame
        
        Returns:
            DataFrame with all results
        """
        if len(self.results) == 0:
            return pd.DataFrame()
        
        return pd.DataFrame(self.results)
    
    def print_summary(self):
        """Print summary statistics of tuning results"""
        if len(self.results) == 0:
            print("No results to summarize")
            return
        
        df = pd.DataFrame(self.results)
        
        print("\n" + "=" * 80)
        print("TUNING SUMMARY")
        print("=" * 80)
        
        print(f"\nTotal configurations tested: {len(df)}")
        print(f"Unique parameter sets: {df[['interval', 'window', 'z_entry', 'z_exit']].drop_duplicates().shape[0]}")
        print(f"Pairs tested: {df[['leader', 'lagger']].drop_duplicates().shape[0]}")
        
        print("\n📊 Performance Distribution:")
        print(df[['sharpe_ratio', 'total_return_pct', 'max_drawdown_pct', 'win_rate', 'num_trades']].describe())
        
        print("\n🏆 Top 5 Configurations by Sharpe Ratio:")
        top5 = df.nlargest(5, 'sharpe_ratio')[
            ['interval', 'window', 'z_entry', 'z_exit', 'leader', 'lagger',
             'sharpe_ratio', 'total_return_pct', 'max_drawdown_pct', 'win_rate', 'num_trades']
        ]
        print(top5.to_string(index=False))
        
        print("\n🎯 Parameter-wise Averages:")
        for param in ['interval', 'window', 'z_entry', 'z_exit']:
            print(f"\n{param.upper()}:")
            grouped = df.groupby(param)['sharpe_ratio'].agg(['mean', 'std', 'count'])
            print(grouped.to_string())


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description='Parameter Tuning for Cross-Asset Lead-Lag Strategy',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--crypto', nargs='+', default=['BTCUSDT', 'ETHUSDT', 'SOLUSDT'],
                       help='Crypto symbols')
    parser.add_argument('--period', type=str, default='7d',
                       help='Data period')
    
    # Tuning space
    parser.add_argument('--intervals', nargs='+', default=['5m', '15m'],
                       help='Intervals to test')
    parser.add_argument('--windows', nargs='+', type=int, default=[30, 60, 90],
                       help='Window sizes to test')
    parser.add_argument('--z-entries', nargs='+', type=float, default=[1.5, 2.0, 2.5],
                       help='Entry Z-score thresholds to test')
    parser.add_argument('--z-exits', nargs='+', type=float, default=[0.3, 0.5],
                       help='Exit Z-score thresholds to test')
    
    # Backtest config
    parser.add_argument('--capital', type=float, default=100000.0,
                       help='Initial capital')
    parser.add_argument('--cost', type=float, default=0.001,
                       help='Transaction cost')
    parser.add_argument('--max-pairs', type=int, default=3,
                       help='Max pairs per interval')
    
    # Output
    parser.add_argument('--output', type=str, default='tuning_results.csv',
                       help='Output CSV file')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("CROSS-ASSET LEAD-LAG PARAMETER TUNER")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    try:
        # Initialize tuner
        tuner = ParameterTuner(
            crypto_symbols=args.crypto,
            equity_symbols={'SP500': '^GSPC', 'NASDAQ': '^IXIC', 'SET50': '^SET.BK'},
            period=args.period,
            initial_capital=args.capital,
            transaction_cost=args.cost,
            min_correlation=0.3,
            max_lag=5
        )
        
        # Run parameter sweep
        tuner.run_parameter_sweep(
            intervals=args.intervals,
            windows=args.windows,
            z_entries=args.z_entries,
            z_exits=args.z_exits,
            max_pairs_per_interval=args.max_pairs
        )
        
        # Save results
        tuner.save_results(args.output)
        
        # Print summary
        tuner.print_summary()
        
        print(f"\n✅ Tuning complete at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 Results saved to: {args.output}")
        print(f"\n💡 Next steps:")
        print(f"   1. Open tuning_analysis.ipynb to visualize results")
        print(f"   2. Review heatmaps and sensitivity charts")
        print(f"   3. Select optimal parameters for production")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
