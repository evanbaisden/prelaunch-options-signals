"""
Trading strategy backtesting and performance evaluation framework.
Implements risk-adjusted performance metrics and strategy optimization.
"""
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import logging

from .options_data import OptionsChain
from .flow_analysis import FlowMetrics
from .regression_framework import RegressionResults
from ..common.types import LaunchEvent


@dataclass
class Trade:
    """Individual trade record."""
    entry_date: date
    exit_date: Optional[date]
    symbol: str
    trade_type: str  # 'long', 'short', 'long_call', 'short_put', etc.
    entry_price: float
    exit_price: Optional[float]
    quantity: int
    
    # Trade metadata
    signal_strength: float
    signal_source: str  # 'options_flow', 'earnings_prediction', etc.
    
    # Performance metrics
    pnl: Optional[float] = None
    return_pct: Optional[float] = None
    days_held: Optional[int] = None
    max_drawdown: Optional[float] = None


@dataclass
class PerformanceMetrics:
    """Portfolio performance metrics."""
    # Return metrics
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    
    # Risk metrics
    max_drawdown: float
    var_95: float  # Value at Risk (95%)
    cvar_95: float  # Conditional Value at Risk (95%)
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float  # Gross profit / Gross loss
    
    # Risk-adjusted metrics
    calmar_ratio: float  # Annual return / Max drawdown
    information_ratio: float
    
    # Benchmark comparison
    beta: Optional[float] = None
    alpha: Optional[float] = None
    tracking_error: Optional[float] = None


@dataclass
class BacktestResults:
    """Complete backtest results."""
    strategy_name: str
    start_date: date
    end_date: date
    initial_capital: float
    final_capital: float
    
    # Performance
    performance_metrics: PerformanceMetrics
    
    # Trade history
    trades: List[Trade]
    portfolio_value: pd.Series
    daily_returns: pd.Series
    
    # Benchmark comparison
    benchmark_returns: Optional[pd.Series] = None
    excess_returns: Optional[pd.Series] = None
    
    # Strategy-specific metrics
    signal_accuracy: Optional[float] = None
    avg_signal_strength: Optional[float] = None


class TradingStrategy:
    """Base class for trading strategies."""
    
    def __init__(self, name: str, config=None):
        from ..config import get_config
        self.name = name
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)
    
    def generate_signals(
        self, 
        options_flow: List[FlowMetrics],
        stock_prices: pd.Series,
        additional_data: Optional[Dict] = None
    ) -> pd.DataFrame:
        """Generate trading signals. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement generate_signals")
    
    def calculate_position_size(
        self, 
        signal_strength: float, 
        portfolio_value: float,
        risk_per_trade: float = 0.02
    ) -> int:
        """Calculate position size based on signal strength and risk management."""
        
        # Simple fixed fractional position sizing
        risk_amount = portfolio_value * risk_per_trade
        
        # Adjust by signal strength (0-1 scale)
        adjusted_risk = risk_amount * abs(signal_strength)
        
        # Convert to shares (simplified - would need actual stock price)
        position_size = int(adjusted_risk / 100)  # Assume $100 per share
        
        return max(1, position_size)


class OptionsFlowStrategy(TradingStrategy):
    """Strategy based on options flow analysis."""
    
    def __init__(self, config=None):
        super().__init__("Options Flow Strategy", config)
        
        # Strategy parameters
        self.put_call_threshold = 0.7  # Bullish signal threshold
        self.iv_skew_threshold = 0.05  # IV skew threshold
        self.volume_percentile_threshold = 0.8  # Volume spike threshold
    
    def generate_signals(
        self, 
        options_flow: List[FlowMetrics],
        stock_prices: pd.Series,
        additional_data: Optional[Dict] = None
    ) -> pd.DataFrame:
        """Generate signals based on options flow analysis."""
        
        signals = []
        
        for flow in options_flow:
            signal_strength = 0
            signal_direction = 'neutral'
            
            # Put/Call ratio signal
            if flow.put_call_volume_ratio < self.put_call_threshold:
                signal_strength += 0.3  # Bullish signal
                signal_direction = 'bullish'
            elif flow.put_call_volume_ratio > (2 - self.put_call_threshold):
                signal_strength -= 0.3  # Bearish signal
                signal_direction = 'bearish'
            
            # IV skew signal
            if abs(flow.iv_skew) > self.iv_skew_threshold:
                if flow.iv_skew < 0:  # Calls more expensive than puts
                    signal_strength += 0.2
                else:  # Puts more expensive than calls
                    signal_strength -= 0.2
            
            # Volume signal (requires historical context)
            total_volume = flow.total_call_volume + flow.total_put_volume
            if total_volume > 0:  # Placeholder for volume percentile
                signal_strength += 0.1
            
            # Net flow direction confirmation
            if flow.net_flow_direction == 'bullish':
                signal_strength += 0.2
            elif flow.net_flow_direction == 'bearish':
                signal_strength -= 0.2
            
            # Normalize signal strength
            signal_strength = max(-1, min(1, signal_strength))
            
            signals.append({
                'date': flow.date,
                'ticker': flow.ticker,
                'signal_strength': signal_strength,
                'signal_direction': signal_direction,
                'put_call_ratio': flow.put_call_volume_ratio,
                'iv_skew': flow.iv_skew,
                'total_volume': total_volume
            })
        
        return pd.DataFrame(signals)


class BacktestEngine:
    """Main backtesting engine."""
    
    def __init__(self, config=None):
        from ..config import get_config
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)
    
    def run_backtest(
        self,
        strategy: TradingStrategy,
        options_data: Dict[str, List[FlowMetrics]],
        stock_data: Dict[str, pd.Series],
        start_date: date,
        end_date: date,
        initial_capital: float = 100000,
        benchmark_returns: Optional[pd.Series] = None
    ) -> BacktestResults:
        """Run complete backtest for a trading strategy."""
        
        # Initialize portfolio
        portfolio_value = initial_capital
        portfolio_history = []
        trades = []
        daily_returns = []
        
        # Generate signals for all stocks
        all_signals = {}
        for ticker, flow_data in options_data.items():
            if ticker in stock_data:
                signals_df = strategy.generate_signals(
                    flow_data, stock_data[ticker]
                )
                all_signals[ticker] = signals_df
        
        # Simulate trading
        current_positions = {}  # ticker -> Trade
        
        # Create date range for simulation
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        for current_date in date_range:
            if current_date.weekday() >= 5:  # Skip weekends
                continue
            
            day_pnl = 0
            
            # Check for new signals
            for ticker, signals_df in all_signals.items():
                if ticker not in stock_data:
                    continue
                
                # Get signal for current date
                day_signals = signals_df[signals_df['date'] == current_date.date()]
                
                if not day_signals.empty:
                    signal = day_signals.iloc[0]
                    signal_strength = signal['signal_strength']
                    
                    # Entry logic
                    if abs(signal_strength) > 0.3 and ticker not in current_positions:
                        # Enter position
                        try:
                            stock_price = self._get_stock_price(stock_data[ticker], current_date)
                            if stock_price > 0:
                                position_size = strategy.calculate_position_size(
                                    signal_strength, portfolio_value
                                )
                                
                                trade = Trade(
                                    entry_date=current_date.date(),
                                    exit_date=None,
                                    symbol=ticker,
                                    trade_type='long' if signal_strength > 0 else 'short',
                                    entry_price=stock_price,
                                    exit_price=None,
                                    quantity=position_size,
                                    signal_strength=signal_strength,
                                    signal_source='options_flow'
                                )
                                
                                current_positions[ticker] = trade
                                
                                # Update portfolio value (subtract transaction costs)
                                transaction_cost = abs(position_size * stock_price * 0.001)  # 0.1% cost
                                portfolio_value -= transaction_cost
                        
                        except Exception as e:
                            self.logger.warning(f"Error entering position for {ticker}: {e}")
            
            # Check existing positions for exit signals or stop losses
            positions_to_close = []
            
            for ticker, trade in current_positions.items():
                try:
                    current_price = self._get_stock_price(stock_data[ticker], current_date)
                    days_held = (current_date.date() - trade.entry_date).days
                    
                    # Exit conditions
                    should_exit = False
                    
                    # Time-based exit (hold for max 10 days)
                    if days_held >= 10:
                        should_exit = True
                    
                    # Stop loss (5% loss)
                    if trade.trade_type == 'long':
                        if current_price < trade.entry_price * 0.95:
                            should_exit = True
                    else:  # short
                        if current_price > trade.entry_price * 1.05:
                            should_exit = True
                    
                    # Take profit (10% gain)
                    if trade.trade_type == 'long':
                        if current_price > trade.entry_price * 1.10:
                            should_exit = True
                    else:  # short
                        if current_price < trade.entry_price * 0.90:
                            should_exit = True
                    
                    if should_exit:
                        # Close position
                        trade.exit_date = current_date.date()
                        trade.exit_price = current_price
                        trade.days_held = days_held
                        
                        # Calculate P&L
                        if trade.trade_type == 'long':
                            trade.pnl = (current_price - trade.entry_price) * trade.quantity
                        else:  # short
                            trade.pnl = (trade.entry_price - current_price) * trade.quantity
                        
                        trade.return_pct = trade.pnl / (trade.entry_price * trade.quantity)
                        
                        # Update portfolio
                        portfolio_value += trade.pnl
                        day_pnl += trade.pnl
                        
                        trades.append(trade)
                        positions_to_close.append(ticker)
                
                except Exception as e:
                    self.logger.warning(f"Error processing position for {ticker}: {e}")
            
            # Remove closed positions
            for ticker in positions_to_close:
                del current_positions[ticker]
            
            # Record portfolio value
            portfolio_history.append({
                'date': current_date.date(),
                'portfolio_value': portfolio_value,
                'day_pnl': day_pnl
            })
            
            # Calculate daily return
            if len(portfolio_history) > 1:
                prev_value = portfolio_history[-2]['portfolio_value']
                daily_return = (portfolio_value - prev_value) / prev_value
                daily_returns.append(daily_return)
            else:
                daily_returns.append(0)
        
        # Close remaining positions at end date
        for ticker, trade in current_positions.items():
            try:
                current_price = self._get_stock_price(stock_data[ticker], pd.Timestamp(end_date))
                trade.exit_date = end_date
                trade.exit_price = current_price
                trade.days_held = (end_date - trade.entry_date).days
                
                if trade.trade_type == 'long':
                    trade.pnl = (current_price - trade.entry_price) * trade.quantity
                else:
                    trade.pnl = (trade.entry_price - current_price) * trade.quantity
                
                trade.return_pct = trade.pnl / (trade.entry_price * trade.quantity)
                portfolio_value += trade.pnl
                trades.append(trade)
            
            except Exception as e:
                self.logger.warning(f"Error closing final position for {ticker}: {e}")
        
        # Calculate performance metrics
        portfolio_df = pd.DataFrame(portfolio_history)
        returns_series = pd.Series(daily_returns, index=pd.date_range(start=start_date, periods=len(daily_returns)))
        
        performance_metrics = self._calculate_performance_metrics(
            returns_series, trades, initial_capital, portfolio_value, benchmark_returns
        )
        
        return BacktestResults(
            strategy_name=strategy.name,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            final_capital=portfolio_value,
            performance_metrics=performance_metrics,
            trades=trades,
            portfolio_value=pd.Series(portfolio_df['portfolio_value'].values, 
                                     index=pd.to_datetime(portfolio_df['date'])),
            daily_returns=returns_series,
            benchmark_returns=benchmark_returns
        )
    
    def _get_stock_price(self, stock_series: pd.Series, date: pd.Timestamp) -> float:
        """Get stock price for a given date."""
        try:
            # Find the closest available date
            available_dates = stock_series.index
            closest_date = min(available_dates, key=lambda x: abs((x - date).days))
            
            # Only use if within 5 days
            if abs((closest_date - date).days) <= 5:
                price_info = stock_series[closest_date]
                
                # Handle different data formats
                if isinstance(price_info, pd.Series):
                    return price_info.get('Adj Close', price_info.iloc[0])
                else:
                    return float(price_info)
            
            return 0
        
        except Exception:
            return 0
    
    def _calculate_performance_metrics(
        self,
        returns: pd.Series,
        trades: List[Trade],
        initial_capital: float,
        final_capital: float,
        benchmark_returns: Optional[pd.Series] = None
    ) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        
        if len(returns) == 0 or returns.std() == 0:
            return self._empty_performance_metrics()
        
        # Return metrics
        total_return = (final_capital - initial_capital) / initial_capital
        trading_days = len(returns)
        annualized_return = (1 + total_return) ** (252 / trading_days) - 1 if trading_days > 0 else 0
        volatility = returns.std() * np.sqrt(252)
        
        # Risk-free rate (assume 2%)
        risk_free_rate = 0.02
        
        # Sharpe ratio
        if volatility > 0:
            sharpe_ratio = (annualized_return - risk_free_rate) / volatility
        else:
            sharpe_ratio = 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_deviation = downside_returns.std() * np.sqrt(252)
            sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        else:
            sortino_ratio = sharpe_ratio
        
        # Drawdown analysis
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        
        # Value at Risk (95%)
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
        
        # Trade statistics
        total_trades = len(trades)
        if total_trades > 0:
            winning_trades = len([t for t in trades if t.pnl and t.pnl > 0])
            losing_trades = len([t for t in trades if t.pnl and t.pnl <= 0])
            win_rate = winning_trades / total_trades
            
            wins = [t.pnl for t in trades if t.pnl and t.pnl > 0]
            losses = [abs(t.pnl) for t in trades if t.pnl and t.pnl <= 0]
            
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            
            gross_profit = sum(wins) if wins else 0
            gross_loss = sum(losses) if losses else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        else:
            winning_trades = losing_trades = 0
            win_rate = avg_win = avg_loss = profit_factor = 0
        
        # Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        # Information ratio (vs benchmark)
        information_ratio = 0
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            excess_returns = returns - benchmark_returns
            tracking_error = excess_returns.std() * np.sqrt(252)
            if tracking_error > 0:
                information_ratio = excess_returns.mean() * np.sqrt(252) / tracking_error
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            var_95=var_95,
            cvar_95=cvar_95,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            calmar_ratio=calmar_ratio,
            information_ratio=information_ratio
        )
    
    def _empty_performance_metrics(self) -> PerformanceMetrics:
        """Return empty performance metrics."""
        return PerformanceMetrics(
            total_return=0,
            annualized_return=0,
            volatility=0,
            sharpe_ratio=0,
            sortino_ratio=0,
            max_drawdown=0,
            var_95=0,
            cvar_95=0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0,
            avg_win=0,
            avg_loss=0,
            profit_factor=0,
            calmar_ratio=0,
            information_ratio=0
        )
    
    def optimize_strategy_parameters(
        self,
        strategy_class: type,
        parameter_grid: Dict[str, List],
        options_data: Dict[str, List[FlowMetrics]],
        stock_data: Dict[str, pd.Series],
        start_date: date,
        end_date: date,
        metric: str = 'sharpe_ratio'
    ) -> Tuple[Dict, BacktestResults]:
        """Optimize strategy parameters using grid search."""
        
        best_params = None
        best_score = -np.inf
        best_results = None
        
        # Generate all parameter combinations
        import itertools
        param_names = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())
        
        for param_combination in itertools.product(*param_values):
            params = dict(zip(param_names, param_combination))
            
            try:
                # Create strategy with parameters
                strategy = strategy_class(**params)
                
                # Run backtest
                results = self.run_backtest(
                    strategy, options_data, stock_data, start_date, end_date
                )
                
                # Get metric score
                score = getattr(results.performance_metrics, metric, 0)
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_results = results
                
                self.logger.info(f"Params {params}: {metric} = {score:.3f}")
            
            except Exception as e:
                self.logger.warning(f"Error with params {params}: {e}")
                continue
        
        return best_params, best_results