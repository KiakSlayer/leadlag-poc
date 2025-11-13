#!/usr/bin/env python3
"""
Kiak's Deliverable: Cointegration + Adaptive β Comparison
=========================================================

This script compares:
1. Original: Correlation-only pair selection + Fixed β
2. Enhanced: Cointegration-filtered pairs + Adaptive β (rolling/EMA)

Generates before/after metrics and visualizations for 2-3 example pairs.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from core.data_fetcher import DataFetcher
from core.correlation_analyzer import CorrelationAnalyzer
from core.crossasset_leadlag_model import CrossAssetLeadLagModel, ModelConfig
from core.backtester import Backtester, BacktestConfig


def print_banner(text: str):
    """Print a nice banner"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")


def run_strategy_comparison(
    prices: pd.DataFrame,
    leader: str,
    lagger: str,
    lag: int,
    window: int = 60,
    initial_capital: float = 100000
) -> Dict:
    """
    Run strategy with three beta methods and compare results

    Args:
        prices: Price DataFrame
        leader: Leader asset name
        lagger: Lagger asset name
        lag: Lead-lag offset
        window: Rolling window size
        initial_capital: Starting capital

    Returns:
        Dictionary with results for each beta method
    """
    results = {}

    beta_methods = [
        ('fixed', 'Fixed β (Original)'),
        ('rolling', 'Rolling β (Kiak Enhancement)'),
        ('ema', 'EMA β (Kiak Enhancement)')
    ]

    for beta_method, description in beta_methods:
        print(f"  Running with {description}...")

        # Configure model
        config = ModelConfig(
            window=window,
            z_entry=2.0,
            z_exit=0.5,
            beta_method=beta_method,
            beta_lookback=60,
            beta_halflife=30
        )

        model = CrossAssetLeadLagModel(config)
        signals = model.run_strategy(prices, leader, lagger, lag)

        if signals.empty:
            print(f"    ⚠️  No signals generated for {beta_method}")
            continue

        # Run backtest
        bt_config = BacktestConfig(
            initial_capital=initial_capital,
            transaction_cost=0.001
        )
        backtester = Backtester(bt_config)
        backtest_results = backtester.run_backtest(signals, prices, leader, lagger)

        metrics = backtest_results['metrics']

        results[beta_method] = {
            'description': description,
            'signals': signals,
            'equity_curve': backtest_results['equity_curve'],
            'trades': backtest_results['trades'],
            'metrics': metrics
        }

        print(f"    ✓ Sharpe: {metrics.get('sharpe_ratio', 0):.2f} | "
              f"Return: {metrics.get('total_return_pct', 0):.2f}% | "
              f"Trades: {metrics.get('num_trades', 0)}")

    return results


def compare_metrics(results: Dict) -> pd.DataFrame:
    """
    Create comparison table of metrics

    Args:
        results: Dictionary of results from run_strategy_comparison

    Returns:
        DataFrame with comparison metrics
    """
    comparison = []

    for method, data in results.items():
        metrics = data['metrics']
        comparison.append({
            'Method': data['description'],
            'Total Return (%)': metrics.get('total_return_pct', 0),
            'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
            'Sortino Ratio': metrics.get('sortino_ratio', 0),
            'Max Drawdown (%)': metrics.get('max_drawdown_pct', 0),
            'Win Rate (%)': metrics.get('win_rate', 0) * 100,
            'Num Trades': metrics.get('num_trades', 0),
            'Final Capital ($)': metrics.get('final_capital', 0)
        })

    df = pd.DataFrame(comparison)

    # Calculate improvements vs fixed
    if 'fixed' in results:
        fixed_sharpe = results['fixed']['metrics'].get('sharpe_ratio', 0)
        for method in results.keys():
            if method != 'fixed':
                method_sharpe = results[method]['metrics'].get('sharpe_ratio', 0)
                improvement = ((method_sharpe - fixed_sharpe) / abs(fixed_sharpe) * 100) if fixed_sharpe != 0 else 0
                print(f"  {method.upper()} vs FIXED: {improvement:+.1f}% Sharpe improvement")

    return df


def plot_comparison(
    results: Dict,
    prices: pd.DataFrame,
    leader: str,
    lagger: str,
    save_path: str = None
):
    """
    Create before/after comparison plots

    Args:
        results: Dictionary of results
        prices: Price DataFrame
        leader: Leader asset name
        lagger: Lagger asset name
        save_path: Path to save figure
    """
    fig = plt.figure(figsize=(20, 12))

    # Plot 1: Beta Evolution
    ax1 = plt.subplot(3, 2, 1)
    for method, data in results.items():
        signals = data['signals']
        if not signals.empty and 'beta' in signals.columns:
            beta_series = signals['beta'].dropna()
            ax1.plot(beta_series.index, beta_series.values,
                    label=data['description'], linewidth=1.5, alpha=0.8)

    ax1.set_title('Beta (β) Evolution Over Time', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Beta (Hedge Ratio)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Spread Evolution
    ax2 = plt.subplot(3, 2, 2)
    for method, data in results.items():
        signals = data['signals']
        if not signals.empty and 'spread' in signals.columns:
            spread_series = signals['spread'].dropna()
            ax2.plot(spread_series.index, spread_series.values,
                    label=data['description'], linewidth=1.5, alpha=0.8)

    ax2.set_title('Spread Evolution', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Spread')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)

    # Plot 3: Z-Score Comparison
    ax3 = plt.subplot(3, 2, 3)
    for method, data in results.items():
        signals = data['signals']
        if not signals.empty and 'zscore' in signals.columns:
            zscore_series = signals['zscore'].dropna()
            ax3.plot(zscore_series.index, zscore_series.values,
                    label=data['description'], linewidth=1.5, alpha=0.8)

    ax3.axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='Entry (+2)')
    ax3.axhline(y=-2.0, color='red', linestyle='--', alpha=0.5, label='Entry (-2)')
    ax3.axhline(y=0.5, color='green', linestyle=':', alpha=0.5, label='Exit (+0.5)')
    ax3.axhline(y=-0.5, color='green', linestyle=':', alpha=0.5, label='Exit (-0.5)')
    ax3.set_title('Z-Score Comparison', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Z-Score')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Equity Curves
    ax4 = plt.subplot(3, 2, 4)
    for method, data in results.items():
        equity_curve = data['equity_curve']
        if not equity_curve.empty:
            equity_series = equity_curve['total_equity']
            ax4.plot(equity_series.index, equity_series.values,
                    label=data['description'], linewidth=2)

    ax4.set_title('Equity Curve Comparison', fontsize=13, fontweight='bold')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Portfolio Value ($)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Returns Distribution
    ax5 = plt.subplot(3, 2, 5)
    for method, data in results.items():
        equity_curve = data['equity_curve']
        if not equity_curve.empty:
            returns = equity_curve['total_equity'].pct_change().dropna() * 100
            ax5.hist(returns, bins=30, alpha=0.5, label=data['description'])

    ax5.set_title('Returns Distribution', fontsize=13, fontweight='bold')
    ax5.set_xlabel('Returns (%)')
    ax5.set_ylabel('Frequency')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Metrics Bar Chart
    ax6 = plt.subplot(3, 2, 6)
    metrics_to_plot = ['sharpe_ratio', 'total_return_pct', 'win_rate']
    x = np.arange(len(metrics_to_plot))
    width = 0.25

    for idx, (method, data) in enumerate(results.items()):
        metrics = data['metrics']
        values = [
            metrics.get('sharpe_ratio', 0),
            metrics.get('total_return_pct', 0),
            metrics.get('win_rate', 0) * 100
        ]
        ax6.bar(x + idx * width, values, width, label=data['description'])

    ax6.set_title('Performance Metrics Comparison', fontsize=13, fontweight='bold')
    ax6.set_xticks(x + width)
    ax6.set_xticklabels(['Sharpe', 'Return (%)', 'Win Rate (%)'])
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'Kiak\'s Enhancement: {leader} → {lagger}',
                fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved comparison plot to {save_path}")

    return fig


def main():
    print_banner("🧩 KIAK'S DELIVERABLE: Cointegration + Adaptive β")

    # Step 1: Fetch Data
    print_banner("STEP 1: Data Collection")
    fetcher = DataFetcher()
    data = fetcher.fetch_all_assets(
        crypto_symbols=['BTCUSDT', 'ETHUSDT', 'SOLUSDT'],
        equity_symbols={'SP500': '^GSPC', 'NASDAQ': '^IXIC'},
        period="7d",
        interval="5m"  # Use 5m for more stable data
    )

    aligned_data = fetcher.align_timestamps(data, method="inner")
    prices = fetcher.get_close_prices(aligned_data)

    if len(prices) < 100:
        print("⚠️  Insufficient data. Exiting.")
        return

    print(f"✓ Collected {len(prices)} timestamps across {len(prices.columns)} assets")

    # Step 2: Correlation Analysis
    print_banner("STEP 2: Correlation Analysis (Original Method)")
    analyzer = CorrelationAnalyzer(prices)

    crypto_assets = [col for col in prices.columns if 'USDT' in col]
    index_assets = [col for col in prices.columns if 'USDT' not in col]

    best_pairs = analyzer.find_best_pairs(crypto_assets, index_assets, min_correlation=0.3)
    print(f"✓ Found {len(best_pairs)} pairs with |corr| >= 0.3")

    # Step 3: Cointegration Filtering
    print_banner("STEP 3: Cointegration Filtering (Kiak's Enhancement)")
    cointegrated_pairs = analyzer.filter_cointegrated_pairs(best_pairs, significance_level=0.05)

    if len(cointegrated_pairs) == 0:
        print("⚠️  No cointegrated pairs found. Using correlation pairs instead.")
        test_pairs = best_pairs[:2]
    else:
        print(f"✓ {len(cointegrated_pairs)} pairs passed cointegration test")
        test_pairs = [(p[0], p[1], p[2]) for p in cointegrated_pairs[:2]]

    # Step 4: Run Comparison on Example Pairs
    all_comparisons = {}

    for leader, lagger, corr in test_pairs[:2]:  # Limit to 2 pairs for demo
        print_banner(f"STEP 4: Testing Pair: {leader} → {lagger}")

        results = run_strategy_comparison(prices, leader, lagger, lag=0, window=60)

        if len(results) == 0:
            print(f"  ⚠️  No valid results for {leader}-{lagger}")
            continue

        # Print comparison metrics
        print("\n  Metrics Comparison:")
        comparison_df = compare_metrics(results)
        print(comparison_df.to_string(index=False))

        # Generate plots
        plot_path = f"kiak_comparison_{leader}_{lagger}.png"
        plot_comparison(results, prices, leader, lagger, save_path=plot_path)

        all_comparisons[f"{leader}-{lagger}"] = {
            'results': results,
            'comparison_df': comparison_df
        }

    # Step 5: Final Summary
    print_banner("STEP 5: Final Summary")

    print("📊 Cointegration Test Results:")
    print(f"  Original pairs (correlation-only): {len(best_pairs)}")
    print(f"  Cointegrated pairs (enhanced): {len(cointegrated_pairs)}")
    print(f"  Filter rate: {(1 - len(cointegrated_pairs)/len(best_pairs))*100:.1f}% pairs removed\n")

    print("📈 Adaptive β Performance:")
    for pair_name, data in all_comparisons.items():
        print(f"\n  {pair_name}:")
        comparison_df = data['comparison_df']
        if 'fixed' in data['results']:
            fixed_sharpe = data['results']['fixed']['metrics'].get('sharpe_ratio', 0)
            for method in ['rolling', 'ema']:
                if method in data['results']:
                    method_sharpe = data['results'][method]['metrics'].get('sharpe_ratio', 0)
                    improvement = ((method_sharpe - fixed_sharpe) / abs(fixed_sharpe) * 100) if fixed_sharpe != 0 else 0
                    status = "✓ PASS" if improvement >= 10 else "✗ FAIL"
                    print(f"    {status} {method.upper()}: {improvement:+.1f}% Sharpe improvement")

    print("\n✅ Kiak's deliverable complete!")
    print("   Files generated: kiak_comparison_*.png")


if __name__ == "__main__":
    main()
