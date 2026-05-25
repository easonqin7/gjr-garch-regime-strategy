"""
GJR-GARCH volatility-regime strategy.

This script downloads equity and benchmark price data, estimates rolling
GJR-GARCH volatility forecasts, classifies market regimes by volatility
percentile, and backtests a simple exposure-control strategy that holds the
asset during low-volatility regimes and moves to cash during high-volatility
regimes.
"""

# Core numerical, data, market-data, modeling, type, plotting, and warning
# libraries used throughout the research pipeline are imported here.
import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model
from dataclasses import dataclass
from typing import Tuple, Dict
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# This output directory stores generated CSV tables and PNG charts for GitHub
# review and resume presentation.
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


# This dataclass centralizes the strategy universe, research sample, GARCH
# window, regime threshold, position sizing, and commission assumptions.
@dataclass
class StrategyConfig:
    ticker: str = "TSLA"
    benchmark_ticker: str = "^GSPC"
    start_date: str = "2010-01-01"
    end_date: str = "2025-12-31"
    garch_window: int = 252
    vol_percentile_threshold: float = 60.0
    low_vol_weight: float = 1.0
    high_vol_weight: float = 0.0
    commission_per_share: float = 0.005
    min_commission: float = 1.0


# This engine downloads asset and benchmark data from Yahoo Finance and returns
# clean price DataFrames for the rest of the pipeline.
class DataEngine:
    def __init__(self, config: StrategyConfig):
        self.config = config
    
    def fetch_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        asset = yf.download(
            self.config.ticker,
            start=self.config.start_date,
            end=self.config.end_date,
            progress=False,
            auto_adjust=False
        )
        benchmark = yf.download(
            self.config.benchmark_ticker,
            start=self.config.start_date,
            end=self.config.end_date,
            progress=False,
            auto_adjust=False
        )
        if isinstance(asset.columns, pd.MultiIndex):
            asset.columns = asset.columns.get_level_values(0)
        if isinstance(benchmark.columns, pd.MultiIndex):
            benchmark.columns = benchmark.columns.get_level_values(0)
        asset = asset[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].dropna()
        benchmark = benchmark[['Adj Close']].dropna()
        benchmark.columns = ['Benchmark_Close']
        return asset, benchmark


# This engine fits GJR-GARCH(1,1,1) models and produces rolling one-step-ahead
# conditional volatility forecasts.
class GJRGARCHEngine:
    def __init__(self, config: StrategyConfig):
        self.config = config
    
    def fit_gjr_garch(self, returns: pd.Series) -> Dict:
        model = arch_model(
            returns * 100,
            vol='Garch',
            p=1,
            o=1,
            q=1,
            dist='t'
        )
        result = model.fit(disp='off', show_warning=False)
        return {
            'model': result,
            'conditional_vol': result.conditional_volatility / 100,
            'params': result.params
        }
    
    def compute_rolling_volatility(self, returns: pd.Series) -> pd.Series:
        n = len(returns)
        window = self.config.garch_window
        conditional_vol = pd.Series(index=returns.index, dtype=float)
        
        for i in range(window, n):
            window_returns = returns.iloc[i-window:i]
            try:
                result = self.fit_gjr_garch(window_returns)
                forecast = result['model'].forecast(horizon=1)
                conditional_vol.iloc[i] = np.sqrt(forecast.variance.values[-1, 0]) / 100
            except Exception:
                conditional_vol.iloc[i] = window_returns.std()
        
        return conditional_vol


# This class converts conditional volatility forecasts into low-volatility and
# high-volatility regimes using an expanding percentile threshold.
class RegimeClassifier:
    def __init__(self, config: StrategyConfig):
        self.config = config
    
    def classify_regime(self, conditional_vol: pd.Series) -> pd.Series:
        expanding_percentile = conditional_vol.expanding(min_periods=20).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100,
            raw=False
        )
        regime = pd.Series(index=conditional_vol.index, dtype=int)
        regime[expanding_percentile <= self.config.vol_percentile_threshold] = 0
        regime[expanding_percentile > self.config.vol_percentile_threshold] = 1
        return regime


# This model estimates trading commission using per-share and minimum-commission
# assumptions.
class CommissionModel:
    def __init__(self, config: StrategyConfig):
        self.config = config
    
    def calculate_commission(self, shares_traded: float, price: float) -> float:
        if shares_traded == 0:
            return 0.0
        commission = abs(shares_traded) * self.config.commission_per_share
        return max(commission, self.config.min_commission)


# This engine converts regimes into target weights and simulates portfolio
# value after commissions.
class StrategyEngine:
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.commission_model = CommissionModel(config)
    
    def generate_signals(self, regime: pd.Series) -> pd.Series:
        target_weight = pd.Series(index=regime.index, dtype=float)
        target_weight[regime == 0] = self.config.low_vol_weight
        target_weight[regime == 1] = self.config.high_vol_weight
        signal = target_weight.shift(1)
        return signal
    
    def run_backtest(
        self,
        prices: pd.DataFrame,
        signals: pd.Series,
        initial_capital: float = 100000.0
    ) -> pd.DataFrame:
        
        aligned_idx = prices.index.intersection(signals.dropna().index)
        prices = prices.loc[aligned_idx].copy()
        signals = signals.loc[aligned_idx].copy()
        
        results = pd.DataFrame(index=aligned_idx)
        results['Close'] = prices['Adj Close']
        results['Signal'] = signals
        results['Returns'] = prices['Adj Close'].pct_change()
        
        portfolio_value = initial_capital
        shares_held = 0.0
        cash = initial_capital
        
        portfolio_values = []
        commissions_paid = []
        
        for i, (date, row) in enumerate(results.iterrows()):
            price = row['Close']
            target_weight = row['Signal']
            
            if pd.isna(target_weight) or pd.isna(price):
                portfolio_values.append(portfolio_value)
                commissions_paid.append(0.0)
                continue
            
            current_equity = shares_held * price + cash
            target_shares = (target_weight * current_equity) / price
            shares_to_trade = target_shares - shares_held
            
            commission = self.commission_model.calculate_commission(shares_to_trade, price)
            
            trade_cost = shares_to_trade * price + commission
            
            if cash - trade_cost >= -current_equity * 0.01:
                cash -= trade_cost
                shares_held = target_shares
            
            portfolio_value = shares_held * price + cash
            portfolio_values.append(portfolio_value)
            commissions_paid.append(commission)
        
        results['Portfolio_Value'] = portfolio_values
        results['Commissions'] = commissions_paid
        results['Strategy_Returns'] = results['Portfolio_Value'].pct_change()
        
        return results


# This engine aligns benchmark returns to the strategy backtest dates.
class BenchmarkEngine:
    def __init__(self, config: StrategyConfig):
        self.config = config
    
    def compute_benchmark_returns(
        self,
        benchmark: pd.DataFrame,
        strategy_index: pd.DatetimeIndex
    ) -> pd.Series:
        aligned_benchmark = benchmark.reindex(strategy_index)
        benchmark_returns = aligned_benchmark['Benchmark_Close'].pct_change()
        return benchmark_returns


# This class calculates performance, risk, exposure, drawdown, and regime
# diagnostics for the strategy and benchmark.
class PerformanceAnalyzer:
    def __init__(self, config: StrategyConfig):
        self.config = config
    
    def compute_coc_equity(self, returns: pd.Series, initial: float = 100.0) -> pd.Series:
        return initial * (1 + returns).cumprod()
    
    def compute_metrics(self, returns: pd.Series, name: str = "Strategy") -> Dict:
        returns_clean = returns.dropna()
        
        total_return = (1 + returns_clean).prod() - 1
        n_years = len(returns_clean) / 252
        cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        
        annual_vol = returns_clean.std() * np.sqrt(252)
        sharpe = cagr / annual_vol if annual_vol > 0 else 0
        
        downside_returns = returns_clean[returns_clean < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino = cagr / downside_vol if downside_vol > 0 else 0
        
        cum_returns = (1 + returns_clean).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = cum_returns / rolling_max - 1
        max_dd = drawdowns.min()
        
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        
        return {
            f'{name}_Total_Return': total_return,
            f'{name}_CAGR': cagr,
            f'{name}_Annual_Vol': annual_vol,
            f'{name}_Sharpe': sharpe,
            f'{name}_Sortino': sortino,
            f'{name}_Max_Drawdown': max_dd,
            f'{name}_Calmar': calmar
        }
    
    def compute_detailed_stats(
        self,
        results: pd.DataFrame,
        regime: pd.Series,
        benchmark_returns: pd.Series
    ) -> Dict:
        signals = results['Signal'].dropna()
        returns = results['Strategy_Returns'].dropna()
        commissions = results['Commissions']
        
        regime_aligned = regime.reindex(signals.index).dropna()
        
        total_days = len(signals)
        days_invested = (signals > 0).sum()
        days_cash = (signals == 0).sum()
        pct_invested = days_invested / total_days * 100
        
        low_vol_days = (regime_aligned == 0).sum()
        high_vol_days = (regime_aligned == 1).sum()
        
        signal_changes = (signals.diff().abs() > 0.5).sum()
        
        total_commissions = commissions.sum()
        
        winning_days = (returns > 0).sum()
        losing_days = (returns < 0).sum()
        win_rate = winning_days / (winning_days + losing_days) * 100 if (winning_days + losing_days) > 0 else 0
        
        avg_win = returns[returns > 0].mean() * 100 if len(returns[returns > 0]) > 0 else 0
        avg_loss = returns[returns < 0].mean() * 100 if len(returns[returns < 0]) > 0 else 0
        
        profit_factor = abs(returns[returns > 0].sum() / returns[returns < 0].sum()) if returns[returns < 0].sum() != 0 else np.inf
        
        bench_clean = benchmark_returns.dropna()
        correlation = returns.corr(bench_clean.reindex(returns.index))
        
        strategy_equity = (1 + returns).cumprod()
        benchmark_equity = (1 + bench_clean).cumprod()
        
        strat_dd = strategy_equity / strategy_equity.expanding().max() - 1
        bench_dd = benchmark_equity / benchmark_equity.expanding().max() - 1
        
        strat_dd_duration = self._max_dd_duration(strat_dd)
        bench_dd_duration = self._max_dd_duration(bench_dd)
        
        returns_invested = returns[signals.shift(1) > 0].dropna()
        returns_cash = returns[signals.shift(1) == 0].dropna()
        
        return {
            'total_trading_days': total_days,
            'days_invested': days_invested,
            'days_in_cash': days_cash,
            'pct_time_invested': pct_invested,
            'low_vol_regime_days': low_vol_days,
            'high_vol_regime_days': high_vol_days,
            'regime_switches': signal_changes,
            'total_commissions': total_commissions,
            'win_rate': win_rate,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'profit_factor': profit_factor,
            'correlation_to_benchmark': correlation,
            'strategy_max_dd_duration': strat_dd_duration,
            'benchmark_max_dd_duration': bench_dd_duration,
            'avg_return_when_invested': returns_invested.mean() * 100 if len(returns_invested) > 0 else 0,
            'avg_return_when_cash': returns_cash.mean() * 100 if len(returns_cash) > 0 else 0
        }
    
    def _max_dd_duration(self, drawdown_series: pd.Series) -> int:
        is_dd = drawdown_series < 0
        groups = (~is_dd).cumsum()
        dd_lengths = is_dd.groupby(groups).sum()
        return int(dd_lengths.max()) if len(dd_lengths) > 0 else 0
    
    def generate_report(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> pd.DataFrame:
        strategy_metrics = self.compute_metrics(strategy_returns, "Strategy")
        benchmark_metrics = self.compute_metrics(benchmark_returns, "Benchmark")
        
        all_metrics = {**strategy_metrics, **benchmark_metrics}
        report = pd.DataFrame.from_dict(all_metrics, orient='index', columns=['Value'])
        return report


# This class creates and saves the main strategy diagnostic chart.
class Visualizer:
    def __init__(self, config: StrategyConfig):
        self.config = config
    
    def plot_results(
        self,
        results: pd.DataFrame,
        benchmark_returns: pd.Series,
        conditional_vol: pd.Series,
        regime: pd.Series,
        performance_report: pd.DataFrame,
        save_path: str = "gjr_garch_regime_results.png"
    ):
        fig, axes = plt.subplots(4, 1, figsize=(14, 16))
        
        strategy_equity = 100 * (1 + results['Strategy_Returns']).cumprod()
        benchmark_equity = 100 * (1 + benchmark_returns).cumprod()
        
        axes[0].plot(strategy_equity.index, strategy_equity.values, label='GJR-GARCH Regime Strategy', linewidth=1.5)
        axes[0].plot(benchmark_equity.index, benchmark_equity.values, label='S&P 500 (Buy & Hold)', linewidth=1.5, alpha=0.7)
        axes[0].set_ylabel('Equity (CoC)')
        axes[0].set_title(f'GJR-GARCH Two-Regime Strategy vs Benchmark')
        axes[0].legend(loc='upper left')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')
        
        vol_aligned = conditional_vol.reindex(results.index).dropna()
        axes[1].plot(vol_aligned.index, vol_aligned.values * 100, color='purple', linewidth=0.8)
        axes[1].set_ylabel('Conditional Volatility (%)')
        axes[1].set_title('GJR-GARCH Conditional Volatility (1-Day Ahead Forecast)')
        axes[1].grid(True, alpha=0.3)
        
        regime_aligned = regime.reindex(results.index).dropna()
        high_vol_periods = regime_aligned[regime_aligned == 1]
        for date in high_vol_periods.index:
            axes[2].axvline(x=date, color='red', alpha=0.02, linewidth=0.5)
        axes[2].plot(results.index, results['Signal'].values, color='green', linewidth=1)
        axes[2].set_ylabel('Position Weight')
        axes[2].set_title(f'Regime Classification (Red = High Vol, Threshold: {self.config.vol_percentile_threshold}th Percentile)')
        axes[2].set_ylim(-0.1, 1.1)
        axes[2].grid(True, alpha=0.3)
        
        strategy_dd = strategy_equity / strategy_equity.expanding().max() - 1
        benchmark_dd = benchmark_equity / benchmark_equity.expanding().max() - 1
        axes[3].fill_between(strategy_dd.index, strategy_dd.values, 0, alpha=0.5, label='Strategy DD')
        axes[3].fill_between(benchmark_dd.index, benchmark_dd.values, 0, alpha=0.3, label='Benchmark DD')
        axes[3].set_ylabel('Drawdown')
        axes[3].set_title('Drawdown Comparison')
        axes[3].legend(loc='lower left')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        full_save_path = OUTPUT_DIR / save_path
        plt.savefig(full_save_path, dpi=300, bbox_inches="tight")
        print(f"Saved chart: {full_save_path}")
        plt.show()
        
        return str(full_save_path)


# This facade wires the individual engines together into one end-to-end research
# workflow.
class GJRGARCHRegimeStrategy:
    def __init__(self, config: StrategyConfig = None):
        self.config = config or StrategyConfig()
        self.data_engine = DataEngine(self.config)
        self.garch_engine = GJRGARCHEngine(self.config)
        self.regime_classifier = RegimeClassifier(self.config)
        self.strategy_engine = StrategyEngine(self.config)
        self.benchmark_engine = BenchmarkEngine(self.config)
        self.performance_analyzer = PerformanceAnalyzer(self.config)
        self.visualizer = Visualizer(self.config)
    
    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, str]:
        asset_data, benchmark_data = self.data_engine.fetch_data()
        
        returns = asset_data['Adj Close'].pct_change().dropna()
        
        conditional_vol = self.garch_engine.compute_rolling_volatility(returns)
        
        regime = self.regime_classifier.classify_regime(conditional_vol)
        
        signals = self.strategy_engine.generate_signals(regime)
        
        results = self.strategy_engine.run_backtest(asset_data, signals)
        
        benchmark_returns = self.benchmark_engine.compute_benchmark_returns(
            benchmark_data,
            results.index
        )
        
        performance_report = self.performance_analyzer.generate_report(
            results['Strategy_Returns'],
            benchmark_returns
        )
        
        detailed_stats = self.performance_analyzer.compute_detailed_stats(
            results,
            regime,
            benchmark_returns
        )
        
        plot_path = self.visualizer.plot_results(
            results,
            benchmark_returns,
            conditional_vol,
            regime,
            performance_report
        )
        
        return results, performance_report, detailed_stats, plot_path


# This helper saves all important research outputs to CSV files so the repository
# contains concrete artifacts after the script runs.
def save_research_outputs(
    results: pd.DataFrame,
    performance_report: pd.DataFrame,
    detailed_stats: Dict,
    plot_path: str
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_path = OUTPUT_DIR / "strategy_results.csv"
    results.to_csv(results_path)
    print(f"Saved table: {results_path}")

    report_path = OUTPUT_DIR / "performance_report.csv"
    performance_report.to_csv(report_path)
    print(f"Saved table: {report_path}")

    stats_path = OUTPUT_DIR / "detailed_stats.csv"
    pd.Series(detailed_stats, name="Value").to_csv(stats_path)
    print(f"Saved table: {stats_path}")

    manifest_path = OUTPUT_DIR / "output_manifest.csv"
    pd.DataFrame([
        {"artifact": "strategy_results.csv", "description": "Daily portfolio values, signals, commissions, and returns"},
        {"artifact": "performance_report.csv", "description": "Strategy and benchmark performance metrics"},
        {"artifact": "detailed_stats.csv", "description": "Regime, trading, win-rate, and drawdown diagnostics"},
        {"artifact": Path(plot_path).name, "description": "Equity, volatility, regime, and drawdown chart"},
    ]).to_csv(manifest_path, index=False)
    print(f"Saved table: {manifest_path}")


# This main block configures and runs the TSLA versus S&P 500 GJR-GARCH regime
# strategy, prints the report, and saves CSV/PNG outputs.
if __name__ == "__main__":
    config = StrategyConfig(
        ticker="TSLA",
        benchmark_ticker="^GSPC",
        start_date="2010-01-01",
        end_date="2025-12-31",
        garch_window=252,
        vol_percentile_threshold=60.0,
        low_vol_weight=1.0,
        high_vol_weight=0.0,
        commission_per_share=0.005,
        min_commission=1.0
    )
    
    strategy = GJRGARCHRegimeStrategy(config)
    results, report, detailed_stats, plot_path = strategy.run()
    save_research_outputs(results, report, detailed_stats, plot_path)
    
    print("\n" + "="*70)
    print("GJR-GARCH TWO-REGIME STRATEGY - COMPREHENSIVE REPORT")
    print("="*70)
    
    print("\n" + "-"*70)
    print("CONFIGURATION")
    print("-"*70)
    print(f"  Ticker:                    {config.ticker}")
    print(f"  Benchmark:                 {config.benchmark_ticker}")
    print(f"  Period:                    {config.start_date} to {config.end_date}")
    print(f"  GARCH Window:              {config.garch_window} days")
    print(f"  Volatility Threshold:      {config.vol_percentile_threshold}th percentile")
    print(f"  Low Vol Weight:            {config.low_vol_weight:.0%}")
    print(f"  High Vol Weight:           {config.high_vol_weight:.0%}")
    print(f"  Commission per Share:      ${config.commission_per_share}")
    print(f"  Minimum Commission:        ${config.min_commission}")
    
    print("\n" + "-"*70)
    print("PERFORMANCE METRICS")
    print("-"*70)
    print(f"{'Metric':<30} {'Strategy':>15} {'Benchmark':>15}")
    print("-"*70)
    
    metrics_pairs = [
        ('Total_Return', 'Total Return'),
        ('CAGR', 'CAGR'),
        ('Annual_Vol', 'Annual Volatility'),
        ('Sharpe', 'Sharpe Ratio'),
        ('Sortino', 'Sortino Ratio'),
        ('Max_Drawdown', 'Max Drawdown'),
        ('Calmar', 'Calmar Ratio')
    ]
    
    for metric_key, metric_name in metrics_pairs:
        strat_val = report.loc[f'Strategy_{metric_key}', 'Value']
        bench_val = report.loc[f'Benchmark_{metric_key}', 'Value']
        if 'Return' in metric_name or 'Drawdown' in metric_name or 'Vol' in metric_name:
            print(f"  {metric_name:<28} {strat_val:>14.2%} {bench_val:>14.2%}")
        else:
            print(f"  {metric_name:<28} {strat_val:>14.3f} {bench_val:>14.3f}")
    
    print("\n" + "-"*70)
    print("REGIME ANALYSIS")
    print("-"*70)
    total_days = detailed_stats['total_trading_days']
    low_vol = detailed_stats['low_vol_regime_days']
    high_vol = detailed_stats['high_vol_regime_days']
    print(f"  Total Trading Days:        {total_days:,}")
    print(f"  Low Volatility Days:       {low_vol:,} ({low_vol/total_days*100:.1f}%)")
    print(f"  High Volatility Days:      {high_vol:,} ({high_vol/total_days*100:.1f}%)")
    print(f"  Regime Switches:           {detailed_stats['regime_switches']}")
    
    print("\n" + "-"*70)
    print("EXPOSURE & TRADING")
    print("-"*70)
    print(f"  Days Invested:             {detailed_stats['days_invested']:,} ({detailed_stats['pct_time_invested']:.1f}%)")
    print(f"  Days in Cash:              {detailed_stats['days_in_cash']:,} ({100-detailed_stats['pct_time_invested']:.1f}%)")
    print(f"  Total Commissions Paid:    ${detailed_stats['total_commissions']:,.2f}")
    
    print("\n" + "-"*70)
    print("TRADE STATISTICS")
    print("-"*70)
    print(f"  Win Rate:                  {detailed_stats['win_rate']:.1f}%")
    print(f"  Average Win:               {detailed_stats['avg_win_pct']:.3f}%")
    print(f"  Average Loss:              {detailed_stats['avg_loss_pct']:.3f}%")
    print(f"  Profit Factor:             {detailed_stats['profit_factor']:.2f}")
    print(f"  Correlation to Benchmark:  {detailed_stats['correlation_to_benchmark']:.3f}")
    
    print("\n" + "-"*70)
    print("DRAWDOWN ANALYSIS")
    print("-"*70)
    print(f"  Strategy Max DD Duration:  {detailed_stats['strategy_max_dd_duration']} days")
    print(f"  Benchmark Max DD Duration: {detailed_stats['benchmark_max_dd_duration']} days")
    
    print("\n" + "-"*70)
    print("REGIME RETURN BREAKDOWN")
    print("-"*70)
    print(f"  Avg Daily Return (Invested):  {detailed_stats['avg_return_when_invested']:.4f}%")
    print(f"  Avg Daily Return (Cash):      {detailed_stats['avg_return_when_cash']:.4f}%")
    
    print("\n" + "="*70)
    print("STRATEGY SUMMARY")
    print("="*70)
    strat_cagr = report.loc['Strategy_CAGR', 'Value']
    bench_cagr = report.loc['Benchmark_CAGR', 'Value']
    strat_dd = report.loc['Strategy_Max_Drawdown', 'Value']
    bench_dd = report.loc['Benchmark_Max_Drawdown', 'Value']
    
    outperform = "OUTPERFORMS" if strat_cagr > bench_cagr else "UNDERPERFORMS"
    risk_adj = "BETTER" if report.loc['Strategy_Sharpe', 'Value'] > report.loc['Benchmark_Sharpe', 'Value'] else "WORSE"
    dd_comp = "LOWER" if abs(strat_dd) < abs(bench_dd) else "HIGHER"
    
    print(f"  Strategy {outperform} benchmark by {(strat_cagr - bench_cagr)*100:.2f}% CAGR")
    print(f"  Risk-adjusted returns (Sharpe) are {risk_adj} than benchmark")
    print(f"  Maximum drawdown is {dd_comp} than benchmark ({strat_dd:.1%} vs {bench_dd:.1%})")
    print(f"  Strategy was invested {detailed_stats['pct_time_invested']:.1f}% of the time")
    print("="*70 + "\n")
