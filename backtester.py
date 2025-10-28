"""
Backtesting Engine for Cross-Asset Lead-Lag Strategy
====================================================
Simulates trading and calculates performance metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    initial_capital: float = 100000.0  # Starting capital in USD
    transaction_cost: float = 0.001  # 0.1% transaction cost per trade
    position_size: float = 1.0  # Fraction of capital per position (0-1)
    max_leverage: float = 2.0  # Maximum leverage allowed


class PerformanceMetrics:
    """Calculate various performance metrics"""

    @staticmethod
    def calculate_returns(equity_curve: pd.Series) -> pd.Series:
        """Calculate returns from equity curve"""
        returns = equity_curve.pct_change().fillna(0)
        return returns

    @staticmethod
    def calculate_cumulative_returns(returns: pd.Series) -> pd.Series:
        """Calculate cumulative returns"""
        cum_returns = (1 + returns).cumprod()
        return cum_returns

    @staticmethod
    def calculate_sharpe_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252 * 390  # Minute data: 252 days * 390 minutes
    ) -> float:
        """
        Calculate annualized Sharpe ratio

        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods in a year

        Returns:
            Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - (risk_free_rate / periods_per_year)
        mean_excess = excess_returns.mean()
        std_excess = excess_returns.std()

        if std_excess == 0:
            return 0.0

        sharpe = (mean_excess / std_excess) * np.sqrt(periods_per_year)
        return sharpe

    @staticmethod
    def calculate_max_drawdown(equity_curve: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
        """
        Calculate maximum drawdown

        Args:
            equity_curve: Series of portfolio values

        Returns:
            Tuple of (max_drawdown, peak_date, trough_date)
        """
        if len(equity_curve) < 2:
            return 0.0, None, None

        # Calculate running maximum
        running_max = equity_curve.expanding().max()

        # Calculate drawdown series
        drawdown = (equity_curve - running_max) / running_max

        # Find maximum drawdown
        max_dd = drawdown.min()

        # Find peak and trough dates
        max_dd_idx = drawdown.idxmin()
        peak_idx = running_max[:max_dd_idx].idxmax()

        return max_dd, peak_idx, max_dd_idx

    @staticmethod
    def calculate_sortino_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252 * 390
    ) -> float:
        """
        Calculate Sortino ratio (focuses on downside volatility)

        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods in a year

        Returns:
            Sortino ratio
        """
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - (risk_free_rate / periods_per_year)
        mean_excess = excess_returns.mean()

        # Only consider negative returns for downside deviation
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = downside_returns.std()

        if downside_std == 0 or len(downside_returns) == 0:
            return 0.0

        sortino = (mean_excess / downside_std) * np.sqrt(periods_per_year)
        return sortino

    @staticmethod
    def calculate_win_rate(trades: pd.DataFrame) -> float:
        """Calculate win rate from trades"""
        if len(trades) == 0:
            return 0.0

        winning_trades = len(trades[trades['pnl'] > 0])
        total_trades = len(trades[trades['pnl'] != 0])

        if total_trades == 0:
            return 0.0

        return winning_trades / total_trades


class Backtester:
    """Backtesting engine for lead-lag strategy"""

    def __init__(self, config: BacktestConfig = BacktestConfig()):
        """
        Initialize backtester

        Args:
            config: Backtesting configuration
        """
        self.config = config
        self.metrics = PerformanceMetrics()

    def execute_trades(
        self,
        signals_df: pd.DataFrame,
        prices: pd.DataFrame,
        leader: str,
        lagger: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Execute trades based on signals

        Args:
            signals_df: DataFrame with signals from model
            prices: DataFrame with asset prices
            leader: Name of leader asset
            lagger: Name of lagger asset

        Returns:
            Tuple of (equity_curve_df, trades_df)
        """
        # Initialize portfolio
        capital = self.config.initial_capital
        position_leader = 0.0  # Current position in leader
        position_lagger = 0.0  # Current position in lagger
        equity = []
        trades = []

        prev_signal = "FLAT"

        for timestamp, row in signals_df.iterrows():
            signal = row['signal']

            # Get current prices
            if timestamp not in prices.index:
                equity.append({
                    'timestamp': timestamp,
                    'capital': capital,
                    'position_value': 0.0,
                    'total_equity': capital
                })
                continue

            price_leader = prices.loc[timestamp, leader]
            price_lagger = prices.loc[timestamp, lagger]

            # Calculate current position value
            position_value = (
                position_leader * price_leader +
                position_lagger * price_lagger
            )

            # Total equity
            total_equity = capital + position_value

            # Execute trades based on signal changes
            if signal != prev_signal and signal != "HOLD":
                # Close existing positions
                if position_leader != 0 or position_lagger != 0:
                    pnl = position_value
                    capital += pnl

                    # Transaction costs
                    cost = abs(position_leader * price_leader) + abs(position_lagger * price_lagger)
                    capital -= cost * self.config.transaction_cost

                    trades.append({
                        'timestamp': timestamp,
                        'action': 'CLOSE',
                        'signal': prev_signal,
                        'pnl': pnl,
                        'capital': capital
                    })

                    position_leader = 0.0
                    position_lagger = 0.0

                # Open new positions
                if signal == "LONG_leader_SHORT_lagger":
                    # Long leader, short lagger
                    position_size_usd = capital * self.config.position_size

                    # Equal notional on both legs
                    position_leader = position_size_usd / price_leader
                    position_lagger = -position_size_usd / price_lagger

                    # Deduct from capital
                    capital -= position_size_usd

                    # Transaction costs
                    cost = 2 * position_size_usd * self.config.transaction_cost
                    capital -= cost

                    trades.append({
                        'timestamp': timestamp,
                        'action': 'OPEN',
                        'signal': signal,
                        'pnl': 0.0,
                        'capital': capital
                    })

                elif signal == "SHORT_leader_LONG_lagger":
                    # Short leader, long lagger
                    position_size_usd = capital * self.config.position_size

                    position_leader = -position_size_usd / price_leader
                    position_lagger = position_size_usd / price_lagger

                    capital -= position_size_usd

                    cost = 2 * position_size_usd * self.config.transaction_cost
                    capital -= cost

                    trades.append({
                        'timestamp': timestamp,
                        'action': 'OPEN',
                        'signal': signal,
                        'pnl': 0.0,
                        'capital': capital
                    })

                elif signal == "FLAT":
                    # Already closed positions above
                    pass

            prev_signal = signal

            # Recalculate position value and equity
            position_value = (
                position_leader * price_leader +
                position_lagger * price_lagger
            )
            total_equity = capital + position_value

            equity.append({
                'timestamp': timestamp,
                'capital': capital,
                'position_value': position_value,
                'total_equity': total_equity,
                'position_leader': position_leader,
                'position_lagger': position_lagger,
                'signal': signal
            })

        equity_df = pd.DataFrame(equity)
        equity_df.set_index('timestamp', inplace=True)

        trades_df = pd.DataFrame(trades)
        if len(trades_df) > 0:
            trades_df.set_index('timestamp', inplace=True)

        return equity_df, trades_df

    def calculate_metrics(
        self,
        equity_df: pd.DataFrame,
        trades_df: pd.DataFrame
    ) -> Dict:
        """
        Calculate comprehensive performance metrics

        Args:
            equity_df: Equity curve DataFrame
            trades_df: Trades DataFrame

        Returns:
            Dictionary of metrics
        """
        if len(equity_df) < 2:
            return {}

        equity_curve = equity_df['total_equity']
        returns = self.metrics.calculate_returns(equity_curve)
        cum_returns = self.metrics.calculate_cumulative_returns(returns)

        # Calculate metrics
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        sharpe = self.metrics.calculate_sharpe_ratio(returns)
        sortino = self.metrics.calculate_sortino_ratio(returns)
        max_dd, peak_date, trough_date = self.metrics.calculate_max_drawdown(equity_curve)

        # Trade metrics
        num_trades = len(trades_df) if len(trades_df) > 0 else 0
        win_rate = self.metrics.calculate_win_rate(trades_df) if len(trades_df) > 0 else 0.0

        if len(trades_df) > 0 and 'pnl' in trades_df.columns:
            avg_pnl = trades_df[trades_df['action'] == 'CLOSE']['pnl'].mean()
            total_pnl = trades_df[trades_df['action'] == 'CLOSE']['pnl'].sum()
        else:
            avg_pnl = 0.0
            total_pnl = 0.0

        metrics = {
            'initial_capital': self.config.initial_capital,
            'final_capital': equity_curve.iloc[-1],
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd,
            'max_drawdown_pct': max_dd * 100,
            'peak_date': peak_date,
            'trough_date': trough_date,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_pnl_per_trade': avg_pnl,
            'total_pnl': total_pnl,
            'volatility_annual': returns.std() * np.sqrt(252 * 390),
        }

        return metrics

    def run_backtest(
        self,
        signals_df: pd.DataFrame,
        prices: pd.DataFrame,
        leader: str,
        lagger: str
    ) -> Dict:
        """
        Run complete backtest

        Args:
            signals_df: Signals from model
            prices: Price data
            leader: Leader asset name
            lagger: Lagger asset name

        Returns:
            Dictionary with equity_df, trades_df, and metrics
        """
        equity_df, trades_df = self.execute_trades(signals_df, prices, leader, lagger)
        metrics = self.calculate_metrics(equity_df, trades_df)

        return {
            'equity_curve': equity_df,
            'trades': trades_df,
            'metrics': metrics
        }


if __name__ == "__main__":
    # Test the backtester
    from data_fetcher import DataFetcher
    from correlation_analyzer import CorrelationAnalyzer
    from crossasset_leadlag_model import CrossAssetLeadLagModel, ModelConfig

    print("=" * 60)
    print("Testing Backtester")
    print("=" * 60)

    # Fetch data
    fetcher = DataFetcher()
    data = fetcher.fetch_all_assets(
        crypto_symbols=['BTCUSDT', 'ETHUSDT'],
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
    bt_config = BacktestConfig(initial_capital=100000, transaction_cost=0.001)
    backtester = Backtester(bt_config)
    results = backtester.run_backtest(signals, prices, 'BTCUSDT', 'SP500')

    # Print metrics
    print("\n" + "=" * 60)
    print("Backtest Results:")
    print("=" * 60)
    for key, value in results['metrics'].items():
        if isinstance(value, float):
            print(f"{key:25s}: {value:>12.4f}")
        else:
            print(f"{key:25s}: {value}")

    print("\nEquity Curve (last 10):")
    print(results['equity_curve'][['capital', 'position_value', 'total_equity']].tail(10))
