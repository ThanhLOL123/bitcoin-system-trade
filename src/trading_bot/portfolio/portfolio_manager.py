import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from ..models.trading_models import Position, AccountInfo, MarketInfo, TradeSetup, OrderResult
from ..monitoring.trading_metrics import TradingMetrics

logger = logging.getLogger(__name__)

class PortfolioManager:
    """Portfolio management and performance tracking"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.metrics = TradingMetrics()
        
        # Portfolio state
        self.initial_balance = 0.0
        self.current_balance = 0.0
        self.total_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        
        # Performance tracking
        self.trades_history = []
        self.daily_returns = []
        self.equity_curve = []
        self.drawdown_history = []
        
        # Risk metrics
        self.max_equity = 0.0
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        
        # Strategy allocation
        self.strategy_allocation = {}
        self.strategy_performance = {}
    
    async def update_state(self, account_info: AccountInfo, 
                          positions: List[Position], 
                          market_info: MarketInfo):
        """Update portfolio state"""
        try:
            self.current_balance = account_info.balance
            self.unrealized_pnl = sum(pos.unrealized_pnl for pos in positions)
            
            current_equity = account_info.equity
            
            # Initialize if first time
            if self.initial_balance == 0.0:
                self.initial_balance = account_info.balance
                self.max_equity = current_equity
            
            # Update max equity and drawdown
            if current_equity > self.max_equity:
                self.max_equity = current_equity
                self.current_drawdown = 0.0
            else:
                self.current_drawdown = (self.max_equity - current_equity) / self.max_equity
                self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
            
            # Update equity curve
            self.equity_curve.append({
                'timestamp': datetime.now(),
                'equity': current_equity,
                'balance': account_info.balance,
                'unrealized_pnl': self.unrealized_pnl
            })
            
            # Keep last 30 days of equity curve
            cutoff_date = datetime.now() - timedelta(days=30)
            self.equity_curve = [
                entry for entry in self.equity_curve 
                if entry['timestamp'] > cutoff_date
            ]
            
        except Exception as e:
            logger.error(f"Error updating portfolio state: {e}")
    
    async def record_trade(self, trade_setup: TradeSetup, order_result: OrderResult):
        """Record executed trade"""
        try:
            trade_record = {
                'timestamp': datetime.now(),
                'ticket': order_result.ticket,
                'symbol': trade_setup.signal.symbol,
                'side': 'LONG' if trade_setup.order_type.value == 'BUY' else 'SHORT',
                'entry_price': order_result.price,
                'volume': order_result.volume,
                'stop_loss': trade_setup.stop_loss,
                'take_profit': trade_setup.take_profit,
                'strategy': trade_setup.signal.model_name,
                'signal_confidence': trade_setup.signal.confidence,
                'risk_amount': trade_setup.risk_amount,
                'expected_return': trade_setup.expected_return,
                'risk_reward_ratio': trade_setup.risk_reward_ratio,
                'status': 'OPEN'
            }
            
            self.trades_history.append(trade_record)
            
            # Update strategy allocation
            strategy = trade_setup.signal.model_name
            if strategy not in self.strategy_allocation:
                self.strategy_allocation[strategy] = 0.0
            
            position_value = order_result.price * order_result.volume
            self.strategy_allocation[strategy] += position_value
            
            logger.info(f"Recorded trade: {trade_record['ticket']} for strategy {strategy}")
            
        except Exception as e:
            logger.error(f"Error recording trade: {e}")
    
    async def record_trade_close(self, ticket: int, close_price: float, realized_pnl: float):
        """Record trade closure"""
        try:
            # Find and update trade record
            for trade in self.trades_history:
                if trade['ticket'] == ticket and trade['status'] == 'OPEN':
                    trade['status'] = 'CLOSED'
                    trade['close_timestamp'] = datetime.now()
                    trade['close_price'] = close_price
                    trade['realized_pnl'] = realized_pnl
                    trade['trade_duration'] = (
                        trade['close_timestamp'] - trade['timestamp']
                    ).total_seconds() / 3600  # hours
                    
                    # Update realized P&L
                    self.realized_pnl += realized_pnl
                    
                    # Update strategy performance
                    strategy = trade['strategy']
                    if strategy not in self.strategy_performance:
                        self.strategy_performance[strategy] = {
                            'trades': 0,
                            'wins': 0,
                            'losses': 0,
                            'total_pnl': 0.0,
                            'win_rate': 0.0
                        }
                    
                    perf = self.strategy_performance[strategy]
                    perf['trades'] += 1
                    perf['total_pnl'] += realized_pnl
                    
                    if realized_pnl > 0:
                        perf['wins'] += 1
                    else:
                        perf['losses'] += 1
                    
                    perf['win_rate'] = perf['wins'] / perf['trades']
                    
                    # Update strategy allocation
                    position_value = trade['entry_price'] * trade['volume']
                    if strategy in self.strategy_allocation:
                        self.strategy_allocation[strategy] -= position_value
                        self.strategy_allocation[strategy] = max(0, self.strategy_allocation[strategy])
                    
                    logger.info(f"Recorded trade close: {ticket}, P&L: {realized_pnl:.2f}")
                    break
                    
        except Exception as e:
            logger.error(f"Error recording trade close: {e}")
    
    async def update_metrics(self, account_info: AccountInfo, positions: List[Position]):
        """Update portfolio metrics"""
        try:
            await self.update_state(account_info, positions, None)
            
            # Calculate daily returns
            if len(self.equity_curve) >= 2:
                today_equity = self.equity_curve[-1]['equity']
                yesterday_equity = self.equity_curve[-2]['equity']
                daily_return = (today_equity - yesterday_equity) / yesterday_equity
                
                self.daily_returns.append({
                    'date': datetime.now().date(),
                    'return': daily_return
                })
                
                # Keep last 252 days (1 year)
                self.daily_returns = self.daily_returns[-252:]
            
            # Record metrics
            self.metrics.record_portfolio_value(account_info.equity)
            self.metrics.record_drawdown(self.current_drawdown)
            self.metrics.record_pnl(self.realized_pnl, self.unrealized_pnl)
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    def get_performance_summary(self) -> Dict:
        """Get portfolio performance summary"""
        try:
            if not self.equity_curve:
                return {"error": "No performance data available"}
            
            # Basic metrics
            total_return = (self.current_balance - self.initial_balance) / self.initial_balance * 100
            
            # Calculate Sharpe ratio
            if len(self.daily_returns) > 30:
                returns = [r['return'] for r in self.daily_returns[-252:]]  # Last year
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                sharpe_ratio = (avg_return * 252) / (std_return * np.sqrt(252)) if std_return > 0 else 0
            else:
                sharpe_ratio = 0
            
            # Win rate calculation
            closed_trades = [t for t in self.trades_history if t['status'] == 'CLOSED']
            if closed_trades:
                winning_trades = [t for t in closed_trades if t['realized_pnl'] > 0]
                win_rate = len(winning_trades) / len(closed_trades) * 100
                avg_win = np.mean([t['realized_pnl'] for t in winning_trades]) if winning_trades else 0
                avg_loss = np.mean([t['realized_pnl'] for t in closed_trades if t['realized_pnl'] < 0])
                profit_factor = abs(avg_win / avg_loss) if avg_loss < 0 else 0
            else:
                win_rate = 0
                avg_win = 0
                avg_loss = 0
                profit_factor = 0
            
            return {
                'total_return_percent': round(total_return, 2),
                'realized_pnl': round(self.realized_pnl, 2),
                'unrealized_pnl': round(self.unrealized_pnl, 2),
                'max_drawdown_percent': round(self.max_drawdown * 100, 2),
                'current_drawdown_percent': round(self.current_drawdown * 100, 2),
                'sharpe_ratio': round(sharpe_ratio, 2),
                'win_rate_percent': round(win_rate, 2),
                'profit_factor': round(profit_factor, 2),
                'total_trades': len(closed_trades),
                'average_win': round(avg_win, 2),
                'average_loss': round(avg_loss, 2),
                'strategy_performance': self.strategy_performance,
                'current_allocation': self.strategy_allocation
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance summary: {e}")
            return {"error": str(e)}