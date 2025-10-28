#!/usr/bin/env python3
"""
Cross-Asset Lead-Lag Arbitrage System - Main PoC Script
========================================================

This is the main orchestration script for the Cross-Asset Lead-Lag
Arbitrage Proof of Concept.

Features:
- Multi-asset data collection (Crypto + Equity Indices)
- Correlation analysis between crypto-index pairs
- Lead-lag relationship detection
- Z-score based trading signals
- Backtesting with performance metrics
- Comprehensive visualization

Author: Quantitative Development Team
Date: 2025
"""

import argparse
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import all modules
from data_fetcher import DataFetcher
from correlation_analyzer import CorrelationAnalyzer
from crossasset_leadlag_model import CrossAssetLeadLagModel, ModelConfig
from backtester import Backtester, BacktestConfig
from visualizer import StrategyVisualizer


def print_banner():
    """Print welcome banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║       Cross-Asset Lead-Lag Arbitrage System (PoC)            ║
    ║                                                               ║
    ║       Crypto-Index Lead-Lag Detection & Trading               ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)
    print(f"    Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 67)


def run_full_analysis(
    crypto_symbols=['BTCUSDT', 'ETHUSDT', 'SOLUSDT'],
    equity_symbols={'SP500': '^GSPC', 'NASDAQ': '^IXIC', 'SET50': '^SET.BK'},
    period='5d',
    interval='1m',
    window=60,
    z_entry=2.0,
    z_exit=0.5,
    min_correlation=0.3,
    max_lag=10,
    initial_capital=100000.0,
    transaction_cost=0.001,
    save_plots=True
):
    """
    Run complete cross-asset lead-lag analysis

    Args:
        crypto_symbols: List of crypto symbols
        equity_symbols: Dict of equity index names and tickers
        period: Data period
        interval: Data interval
        window: Rolling window for statistics
        z_entry: Z-score entry threshold
        z_exit: Z-score exit threshold
        min_correlation: Minimum correlation for pair selection
        max_lag: Maximum lag periods to test
        initial_capital: Starting capital for backtest
        transaction_cost: Transaction cost per trade
        save_plots: Whether to save visualization plots

    Returns:
        Dictionary with all results
    """

    print("\n📊 STEP 1: DATA COLLECTION")
    print("=" * 67)

    # Fetch data
    fetcher = DataFetcher()
    data = fetcher.fetch_all_assets(
        crypto_symbols=crypto_symbols,
        equity_symbols=equity_symbols,
        period=period,
        interval=interval
    )

    if len(data) == 0:
        print("✗ No data fetched. Exiting.")
        return None

    # Align timestamps
    aligned_data = fetcher.align_timestamps(data, method="inner")
    prices = fetcher.get_close_prices(aligned_data)

    print(f"\n✓ Data collection complete")
    print(f"  - Assets: {len(prices.columns)}")
    print(f"  - Time points: {len(prices)}")

    if len(prices) == 0:
        print("  - Date range: No data available")
        print("\n✗ Error: No common timestamps found after alignment!")
        print("\n💡 Suggestions:")
        print("   1. Try using a longer time period (--period 7d or --period 1mo)")
        print("   2. Use a larger interval (--interval 5m or --interval 15m)")
        print("   3. Use crypto-only analysis (remove --equity symbols)")
        print("   4. Check if market is currently open for equity indices")
        return None
    else:
        print(f"  - Date range: {prices.index.min()} to {prices.index.max()}")

    # ----------------------------------------------------------------
    print("\n\n🔗 STEP 2: CORRELATION ANALYSIS")
    print("=" * 67)

    analyzer = CorrelationAnalyzer(prices)

    # Get correlation matrix
    corr_matrix = analyzer.calculate_correlation_matrix()
    print("\nCorrelation Matrix:")
    print(corr_matrix)

    # Find best pairs
    crypto_assets = [col for col in prices.columns if 'USDT' in col or 'BTC' in col]
    index_assets = [col for col in prices.columns if col not in crypto_assets]

    print(f"\nCrypto Assets: {crypto_assets}")
    print(f"Index Assets: {index_assets}")

    best_pairs = analyzer.find_best_pairs(
        crypto_assets,
        index_assets,
        min_correlation=min_correlation
    )

    print(f"\n✓ Found {len(best_pairs)} crypto-index pairs with |corr| >= {min_correlation}")

    if len(best_pairs) == 0:
        print("\n⚠️  No pairs found with sufficient correlation!")
        print("\n💡 Suggestions:")
        print(f"   1. Lower the correlation threshold (current: {min_correlation})")
        print("   2. Ensure you have overlapping data between crypto and indices")
        print("   3. Try crypto-crypto pairs instead of crypto-index pairs")
        print("\nExiting analysis...")
        return None

    print("\nTop Correlated Pairs:")
    for crypto, index, corr in best_pairs[:5]:
        print(f"  {crypto:15s} <-> {index:10s}: {corr:+.4f}")

    # Lead-lag analysis
    print(f"\nAnalyzing lead-lag relationships (max_lag={max_lag})...")
    lead_lag_analysis = analyzer.analyze_all_pairs(best_pairs[:5], max_lag=max_lag)

    print("\n✓ Lead-Lag Analysis Results:")
    if len(lead_lag_analysis) > 0:
        print(lead_lag_analysis.to_string(index=False))
    else:
        print("  (No results)")

    if len(lead_lag_analysis) == 0:
        print("\n⚠️  No pairs available for strategy execution.")
        print("Exiting analysis...")
        return None

    # ----------------------------------------------------------------
    print("\n\n📈 STEP 3: STRATEGY EXECUTION")
    print("=" * 67)

    # Prepare model config
    config = ModelConfig(
        window=window,
        z_entry=z_entry,
        z_exit=z_exit
    )

    model = CrossAssetLeadLagModel(config)

    # Store results for all pairs
    all_results = {}

    # Run strategy on top 3 pairs
    for idx, row in lead_lag_analysis.head(3).iterrows():
        leader = row['leader']
        lagger = row['lagger']
        lag = int(row['lag_periods'])
        pair_name = f"{leader}-{lagger}"

        print(f"\n{'─' * 67}")
        print(f"Pair: {pair_name}")
        print(f"Lead-Lag: {row['relationship']}")
        print(f"Correlation: {row['max_correlation']:.4f}")
        print(f"{'─' * 67}")

        # Run model
        print(f"\n  Running Z-score model...")
        signals = model.run_strategy(prices, leader, lagger, lag)

        # Check if signals were generated
        if signals.empty or len(signals) == 0:
            print(f"\n  ⚠️  No signals generated for this pair (insufficient data)")
            print(f"  Skipping to next pair...\n")
            continue

        # Signal summary
        signal_counts = signals['signal'].value_counts()
        print(f"\n  Signal Distribution:")
        for sig, count in signal_counts.items():
            print(f"    {sig:25s}: {count:6d}")

        # ----------------------------------------------------------------
        print(f"\n  Running backtest...")
        bt_config = BacktestConfig(
            initial_capital=initial_capital,
            transaction_cost=transaction_cost,
            position_size=1.0
        )
        backtester = Backtester(bt_config)

        backtest_results = backtester.run_backtest(signals, prices, leader, lagger)
        metrics = backtest_results['metrics']

        print(f"\n  ✓ Backtest Results:")
        print(f"    {'─' * 60}")
        print(f"    {'Total Return:':<30} {metrics.get('total_return_pct', 0):>10.2f}%")
        print(f"    {'Sharpe Ratio:':<30} {metrics.get('sharpe_ratio', 0):>10.2f}")
        print(f"    {'Sortino Ratio:':<30} {metrics.get('sortino_ratio', 0):>10.2f}")
        print(f"    {'Max Drawdown:':<30} {metrics.get('max_drawdown_pct', 0):>10.2f}%")
        print(f"    {'Number of Trades:':<30} {metrics.get('num_trades', 0):>10d}")
        print(f"    {'Win Rate:':<30} {metrics.get('win_rate', 0)*100:>10.2f}%")
        print(f"    {'Final Capital:':<30} ${metrics.get('final_capital', 0):>10,.2f}")
        print(f"    {'─' * 60}")

        # Store results
        all_results[pair_name] = {
            'leader': leader,
            'lagger': lagger,
            'lag': lag,
            'signals': signals,
            'backtest': backtest_results,
            'metrics': metrics
        }

    # Check if any results were generated
    if len(all_results) == 0:
        print("\n\n⚠️  No valid results generated for any pairs!")
        print("   All pairs had insufficient data for analysis.")
        print("\n💡 Suggestions:")
        print("   1. Use a longer time period (--period 7d or --period 1mo)")
        print("   2. Use a larger interval (--interval 5m or --interval 15m)")
        print("   3. Reduce the window size (--window 30)")
        print("   4. Try crypto-crypto pairs instead")
        return None

    # ----------------------------------------------------------------
    print("\n\n📊 STEP 4: VISUALIZATION")
    print("=" * 67)

    visualizer = StrategyVisualizer()

    for pair_name, result in all_results.items():
        print(f"\nGenerating report for {pair_name}...")

        save_path = f"report_{pair_name.replace('-', '_')}.png" if save_plots else None

        fig = visualizer.create_comprehensive_report(
            prices=prices,
            signals_df=result['signals'],
            equity_df=result['backtest']['equity_curve'],
            metrics=result['metrics'],
            leader=result['leader'],
            lagger=result['lagger'],
            save_path=save_path
        )

        if save_plots:
            print(f"  ✓ Saved to {save_path}")

    # ----------------------------------------------------------------
    print("\n\n✅ ANALYSIS COMPLETE")
    print("=" * 67)
    print(f"\nTotal pairs analyzed: {len(all_results)}")
    print(f"Reports generated: {len(all_results)}")

    # Summary table
    print("\n📋 SUMMARY TABLE:")
    print("=" * 67)
    print(f"{'Pair':<25} {'Return %':>10} {'Sharpe':>8} {'MaxDD %':>10} {'Trades':>8}")
    print("─" * 67)

    for pair_name, result in all_results.items():
        m = result['metrics']
        print(f"{pair_name:<25} {m.get('total_return_pct', 0):>10.2f} "
              f"{m.get('sharpe_ratio', 0):>8.2f} {m.get('max_drawdown_pct', 0):>10.2f} "
              f"{m.get('num_trades', 0):>8d}")

    print("=" * 67)

    return {
        'prices': prices,
        'correlations': corr_matrix,
        'lead_lag_analysis': lead_lag_analysis,
        'results': all_results
    }


def main():
    """Main entry point"""

    parser = argparse.ArgumentParser(
        description='Cross-Asset Lead-Lag Arbitrage System PoC',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data arguments
    parser.add_argument('--crypto', nargs='+', default=['BTCUSDT', 'ETHUSDT', 'SOLUSDT'],
                       help='Crypto symbols')
    parser.add_argument('--period', type=str, default='5d',
                       help='Data period (1d, 5d, 1mo, etc.)')
    parser.add_argument('--interval', type=str, default='1m',
                       help='Data interval (1m, 5m, 1h, etc.)')

    # Model arguments
    parser.add_argument('--window', type=int, default=60,
                       help='Rolling window size')
    parser.add_argument('--z-entry', type=float, default=2.0,
                       help='Z-score entry threshold')
    parser.add_argument('--z-exit', type=float, default=0.5,
                       help='Z-score exit threshold')
    parser.add_argument('--min-corr', type=float, default=0.3,
                       help='Minimum correlation threshold')
    parser.add_argument('--max-lag', type=int, default=10,
                       help='Maximum lag periods')

    # Backtest arguments
    parser.add_argument('--capital', type=float, default=100000.0,
                       help='Initial capital')
    parser.add_argument('--cost', type=float, default=0.001,
                       help='Transaction cost (fraction)')

    # Output arguments
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save plots')

    args = parser.parse_args()

    print_banner()

    try:
        results = run_full_analysis(
            crypto_symbols=args.crypto,
            equity_symbols={'SP500': '^GSPC', 'NASDAQ': '^IXIC', 'SET50': '^SET.BK'},
            period=args.period,
            interval=args.interval,
            window=args.window,
            z_entry=args.z_entry,
            z_exit=args.z_exit,
            min_correlation=args.min_corr,
            max_lag=args.max_lag,
            initial_capital=args.capital,
            transaction_cost=args.cost,
            save_plots=not args.no_save
        )

        if results is None:
            print("\n✗ Analysis failed")
            sys.exit(1)

        print(f"\n✅ SUCCESS! Analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

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
