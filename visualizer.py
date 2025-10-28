"""
Visualization Module for Cross-Asset Lead-Lag System
===================================================
Creates comprehensive charts for strategy analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class StrategyVisualizer:
    """Visualization tools for lead-lag strategy results"""

    def __init__(self, figsize: tuple = (16, 12), style: str = 'seaborn-v0_8-darkgrid'):
        """
        Initialize visualizer

        Args:
            figsize: Default figure size
            style: Matplotlib style
        """
        self.figsize = figsize
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')

    def plot_price_series(
        self,
        prices: pd.DataFrame,
        assets: List[str],
        title: str = "Asset Prices",
        normalize: bool = True,
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """
        Plot price series for multiple assets

        Args:
            prices: DataFrame with price data
            assets: List of asset names to plot
            title: Plot title
            normalize: Whether to normalize prices to 100
            ax: Matplotlib axes (creates new if None)

        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)

        for asset in assets:
            if asset in prices.columns:
                series = prices[asset].dropna()

                if normalize:
                    series = 100 * series / series.iloc[0]

                ax.plot(series.index, series.values, label=asset, linewidth=1.5)

        ax.set_xlabel('Time', fontsize=11)
        ax.set_ylabel('Normalized Price (100 = Start)' if normalize else 'Price', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        return ax

    def plot_zscore(
        self,
        signals_df: pd.DataFrame,
        z_entry: float = 2.0,
        z_exit: float = 0.5,
        title: str = "Z-Score Evolution",
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """
        Plot Z-score over time with entry/exit thresholds

        Args:
            signals_df: DataFrame with zscore column
            z_entry: Entry threshold
            z_exit: Exit threshold
            title: Plot title
            ax: Matplotlib axes

        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)

        zscore = signals_df['zscore'].dropna()

        # Plot Z-score
        ax.plot(zscore.index, zscore.values, color='black', linewidth=1.5, label='Z-Score')

        # Plot thresholds
        ax.axhline(y=z_entry, color='red', linestyle='--', linewidth=1.5, label=f'Entry (+{z_entry})')
        ax.axhline(y=-z_entry, color='red', linestyle='--', linewidth=1.5, label=f'Entry (-{z_entry})')
        ax.axhline(y=z_exit, color='green', linestyle=':', linewidth=1.5, label=f'Exit (+{z_exit})')
        ax.axhline(y=-z_exit, color='green', linestyle=':', linewidth=1.5, label=f'Exit (-{z_exit})')
        ax.axhline(y=0, color='blue', linestyle='-', linewidth=0.8, alpha=0.5)

        # Shade extreme zones
        ax.fill_between(zscore.index, z_entry, 5, alpha=0.1, color='red')
        ax.fill_between(zscore.index, -z_entry, -5, alpha=0.1, color='red')

        ax.set_xlabel('Time', fontsize=11)
        ax.set_ylabel('Z-Score', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        return ax

    def plot_equity_curve(
        self,
        equity_df: pd.DataFrame,
        initial_capital: float,
        title: str = "Equity Curve",
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """
        Plot equity curve and drawdown

        Args:
            equity_df: DataFrame with equity data
            initial_capital: Starting capital
            title: Plot title
            ax: Matplotlib axes

        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)

        equity = equity_df['total_equity']

        # Plot equity curve
        ax.plot(equity.index, equity.values, color='blue', linewidth=2, label='Portfolio Value')

        # Plot buy & hold baseline
        ax.axhline(y=initial_capital, color='gray', linestyle='--', linewidth=1.5, label='Initial Capital')

        # Calculate drawdown
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max * 100

        # Create twin axis for drawdown
        ax2 = ax.twinx()
        ax2.fill_between(drawdown.index, 0, drawdown.values, alpha=0.3, color='red', label='Drawdown %')
        ax2.set_ylabel('Drawdown (%)', fontsize=11)
        ax2.set_ylim([drawdown.min() * 1.2, 5])

        ax.set_xlabel('Time', fontsize=11)
        ax.set_ylabel('Portfolio Value ($)', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')

        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=10)

        ax.grid(True, alpha=0.3)

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        return ax

    def plot_returns_distribution(
        self,
        equity_df: pd.DataFrame,
        title: str = "Returns Distribution",
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """
        Plot histogram of returns

        Args:
            equity_df: DataFrame with equity data
            title: Plot title
            ax: Matplotlib axes

        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        returns = equity_df['total_equity'].pct_change().dropna() * 100

        # Plot histogram
        ax.hist(returns, bins=50, alpha=0.7, color='blue', edgecolor='black')

        # Plot mean and median
        mean_ret = returns.mean()
        median_ret = returns.median()

        ax.axvline(mean_ret, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_ret:.4f}%')
        ax.axvline(median_ret, color='green', linestyle='--', linewidth=2, label=f'Median: {median_ret:.4f}%')

        ax.set_xlabel('Returns (%)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        return ax

    def plot_signal_timeline(
        self,
        signals_df: pd.DataFrame,
        prices: pd.DataFrame,
        leader: str,
        lagger: str,
        title: str = "Trading Signals Timeline",
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """
        Plot price with signal markers

        Args:
            signals_df: DataFrame with signals
            prices: DataFrame with prices
            leader: Leader asset name
            lagger: Lagger asset name
            title: Plot title
            ax: Matplotlib axes

        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)

        # Normalize prices
        leader_price = prices[leader].dropna()
        leader_norm = 100 * leader_price / leader_price.iloc[0]

        ax.plot(leader_norm.index, leader_norm.values, label=leader, linewidth=1.5, color='blue')

        # Mark signals
        signals = signals_df['signal']

        long_leader = signals_df[signals == 'LONG_leader_SHORT_lagger'].index
        short_leader = signals_df[signals == 'SHORT_leader_LONG_lagger'].index
        flat = signals_df[signals == 'FLAT'].index

        for ts in long_leader:
            if ts in leader_norm.index:
                ax.scatter(ts, leader_norm.loc[ts], color='green', marker='^', s=100, zorder=5)

        for ts in short_leader:
            if ts in leader_norm.index:
                ax.scatter(ts, leader_norm.loc[ts], color='red', marker='v', s=100, zorder=5)

        for ts in flat:
            if ts in leader_norm.index:
                ax.scatter(ts, leader_norm.loc[ts], color='orange', marker='x', s=100, zorder=5)

        # Create legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markersize=10, label='LONG Leader'),
            Line2D([0], [0], marker='v', color='w', markerfacecolor='red', markersize=10, label='SHORT Leader'),
            Line2D([0], [0], marker='x', color='w', markerfacecolor='orange', markersize=10, label='FLAT'),
        ]
        ax.legend(handles=legend_elements, loc='best', fontsize=10)

        ax.set_xlabel('Time', fontsize=11)
        ax.set_ylabel('Normalized Price', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        return ax

    def create_comprehensive_report(
        self,
        prices: pd.DataFrame,
        signals_df: pd.DataFrame,
        equity_df: pd.DataFrame,
        metrics: Dict,
        leader: str,
        lagger: str,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create comprehensive visualization report

        Args:
            prices: Price data
            signals_df: Signals data
            equity_df: Equity curve data
            metrics: Performance metrics
            leader: Leader asset name
            lagger: Lagger asset name
            save_path: Path to save figure (None = show only)

        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(20, 14))
        gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Price Series
        ax1 = fig.add_subplot(gs[0, :])
        self.plot_price_series(prices, [leader, lagger],
                              title=f"Asset Prices: {leader} vs {lagger}",
                              normalize=True, ax=ax1)

        # 2. Z-Score
        ax2 = fig.add_subplot(gs[1, :])
        self.plot_zscore(signals_df, title="Z-Score with Entry/Exit Thresholds", ax=ax2)

        # 3. Equity Curve
        ax3 = fig.add_subplot(gs[2, :])
        self.plot_equity_curve(equity_df, metrics.get('initial_capital', 100000),
                              title="Portfolio Equity Curve & Drawdown", ax=ax3)

        # 4. Signal Timeline
        ax4 = fig.add_subplot(gs[3, 0])
        self.plot_signal_timeline(signals_df, prices, leader, lagger,
                                 title="Trading Signals on Leader Price", ax=ax4)

        # 5. Returns Distribution
        ax5 = fig.add_subplot(gs[3, 1])
        self.plot_returns_distribution(equity_df, title="Returns Distribution", ax=ax5)

        # Add metrics text box
        metrics_text = f"""
        PERFORMANCE METRICS
        {'='*40}
        Total Return: {metrics.get('total_return_pct', 0):.2f}%
        Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}
        Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}
        Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%

        Number of Trades: {metrics.get('num_trades', 0)}
        Win Rate: {metrics.get('win_rate', 0)*100:.2f}%
        Avg PnL/Trade: ${metrics.get('avg_pnl_per_trade', 0):.2f}

        Initial Capital: ${metrics.get('initial_capital', 0):,.2f}
        Final Capital: ${metrics.get('final_capital', 0):,.2f}
        """

        fig.text(0.02, 0.98, metrics_text, fontsize=10, family='monospace',
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        fig.suptitle(f"Cross-Asset Lead-Lag Strategy: {leader} → {lagger}",
                    fontsize=16, fontweight='bold', y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved visualization to {save_path}")

        return fig


if __name__ == "__main__":
    # Test the visualizer
    from data_fetcher import DataFetcher
    from crossasset_leadlag_model import CrossAssetLeadLagModel, ModelConfig
    from backtester import Backtester, BacktestConfig

    print("=" * 60)
    print("Testing Visualizer")
    print("=" * 60)

    # Fetch data
    fetcher = DataFetcher()
    data = fetcher.fetch_all_assets(
        crypto_symbols=['BTCUSDT'],
        equity_symbols={'SP500': '^GSPC'},
        period="5d",
        interval="1m"
    )

    aligned_data = fetcher.align_timestamps(data, method="inner")
    prices = fetcher.get_close_prices(aligned_data)

    # Run model
    config = ModelConfig(window=60, z_entry=2.0, z_exit=0.5)
    model = CrossAssetLeadLagModel(config)
    signals = model.run_strategy(prices, 'BTCUSDT', 'SP500', lag=0)

    # Run backtest
    bt_config = BacktestConfig(initial_capital=100000)
    backtester = Backtester(bt_config)
    results = backtester.run_backtest(signals, prices, 'BTCUSDT', 'SP500')

    # Create visualization
    visualizer = StrategyVisualizer()
    fig = visualizer.create_comprehensive_report(
        prices=prices,
        signals_df=signals,
        equity_df=results['equity_curve'],
        metrics=results['metrics'],
        leader='BTCUSDT',
        lagger='SP500',
        save_path='leadlag_report.png'
    )

    plt.show()
