import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from ..models.trading_models import TradeSignal, TradeSetup, Position, AccountInfo, MarketInfo
from ..monitoring.trading_metrics import TradingMetrics

logger = logging.getLogger(__name__)

class RiskManager:
    """Comprehensive risk management system"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.metrics = TradingMetrics()
        
        # Risk parameters
        self.max_daily_loss = config.get('max_daily_loss_percent', 2.0)  # 2% max daily loss
        self.max_position_size = config.get('max_position_size_percent', 5.0)  # 5% max per position
        self.max_total_exposure = config.get('max_total_exposure_percent', 20.0)  # 20% max total exposure
        self.max_correlation_exposure = config.get('max_correlation_exposure', 0.7)  # Max correlation
        self.max_drawdown = config.get('max_drawdown_percent', 10.0)  # 10% max drawdown
        
        # Position sizing
        self.base_risk_per_trade = config.get('base_risk_per_trade_percent', 1.0)  # 1% risk per trade
        self.volatility_adjustment = config.get('volatility_adjustment', True)
        self.confidence_adjustment = config.get('confidence_adjustment', True)
        
        # Tracking
        self.daily_pnl = 0.0
        self.total_exposure = 0.0
        self.max_historical_equity = 0.0
        self.trade_history = []
        
    async def validate_trade_signal(self, signal: TradeSignal, 
                                  account_info: AccountInfo,
                                  current_positions: List[Position]) -> TradeSetup:
        """Validate and create trade setup from signal"""
        
        try:
            # Basic signal validation
            if not self._validate_signal_quality(signal):
                raise ValueError("Signal quality insufficient")
            
            # Risk checks
            if not self._check_account_risk(account_info):
                raise ValueError("Account risk limits exceeded")
            
            if not self._check_exposure_limits(current_positions, signal):
                raise ValueError("Exposure limits exceeded")
            
            if not self._check_timing_constraints(signal):
                raise ValueError("Timing constraints not met")
            
            # Calculate position size
            position_size = await self._calculate_position_size(
                signal, account_info, current_positions
            )
            
            if position_size <= 0:
                raise ValueError("No valid position size calculated")
            
            # Create trade setup
            trade_setup = await self._create_trade_setup(
                signal, position_size, account_info
            )
            
            # Final validation
            trade_setup.risk_approved = self._validate_risk_metrics(trade_setup)
            trade_setup.size_approved = self._validate_position_size(trade_setup, account_info)
            trade_setup.timing_approved = self._validate_timing(trade_setup)
            
            if not all([trade_setup.risk_approved, trade_setup.size_approved, trade_setup.timing_approved]):
                raise ValueError("Trade setup validation failed")
            
            logger.info(f"Trade setup validated: {signal.signal_type.value} "
                       f"{position_size} lots, Risk: {trade_setup.risk_amount:.2f}")
            
            return trade_setup
            
        except Exception as e:
            logger.warning(f"Trade validation failed: {e}")
            raise
    
    def _validate_signal_quality(self, signal: TradeSignal) -> bool:
        """Validate signal quality metrics"""
        
        # Minimum confidence threshold
        if signal.confidence < self.config.get('min_signal_confidence', 0.6):
            logger.debug(f"Signal confidence too low: {signal.confidence}")
            return False
        
        # Model confidence threshold
        if signal.model_confidence < self.config.get('min_model_confidence', 0.7):
            logger.debug(f"Model confidence too low: {signal.model_confidence}")
            return False
        
        # Price change threshold
        min_price_change = self.config.get('min_price_change_percent', 0.5)
        if abs(signal.price_change_percent) < min_price_change:
            logger.debug(f"Price change too small: {signal.price_change_percent}%")
            return False
        
        # Risk-reward ratio threshold
        if signal.risk_reward_ratio and signal.risk_reward_ratio < 1.5:
            logger.debug(f"Risk-reward ratio too low: {signal.risk_reward_ratio}")
            return False
        
        return True
    
    async def _check_account_risk(self, account_info: AccountInfo) -> bool:
        """Check account-level risk constraints"""
        
        # Update max historical equity
        self.max_historical_equity = max(self.max_historical_equity, account_info.equity)
        
        # Check drawdown
        if self.max_historical_equity > 0:
            current_drawdown = (self.max_historical_equity - account_info.equity) / self.max_historical_equity * 100
            if current_drawdown > self.max_drawdown:
                logger.warning(f"Max drawdown exceeded: {current_drawdown:.2f}%")
                return False
        
        # Check daily loss
        daily_loss_percent = abs(self.daily_pnl) / account_info.balance * 100
        if self.daily_pnl < 0 and daily_loss_percent > self.max_daily_loss:
            logger.warning(f"Daily loss limit exceeded: {daily_loss_percent:.2f}%")
            return False
        
        # Check margin level
        min_margin_level = self.config.get('min_margin_level', 200.0)
        if account_info.margin_level < min_margin_level:
            logger.warning(f"Margin level too low: {account_info.margin_level:.1f}%")
            return False
        
        return True
    
    def _check_exposure_limits(self, current_positions: List[Position], signal: TradeSignal) -> bool:
        """Check position exposure limits"""
        
        # Calculate current exposure
        current_exposure = sum(abs(pos.volume * pos.current_price) for pos in current_positions)
        
        # Check if adding new position would exceed limits
        estimated_position_value = signal.max_position_size * signal.current_price
        total_exposure_percent = (current_exposure + estimated_position_value) / self.max_historical_equity * 100
        
        if total_exposure_percent > self.max_total_exposure:
            logger.warning(f"Total exposure limit exceeded: {total_exposure_percent:.2f}%")
            return False
        
        # Check position concentration
        symbol_exposure = sum(
            abs(pos.volume * pos.current_price) 
            for pos in current_positions 
            if pos.symbol == signal.symbol
        )
        
        symbol_exposure_percent = symbol_exposure / self.max_historical_equity * 100
        if symbol_exposure_percent > self.max_position_size:
            logger.warning(f"Symbol exposure limit exceeded: {symbol_exposure_percent:.2f}%")
            return False
        
        return True
    
    def _check_timing_constraints(self, signal: TradeSignal) -> bool:
        """Check timing constraints for trading"""
        
        current_time = datetime.now()
        
        # Check market hours (if applicable)
        market_hours = self.config.get('market_hours')
        if market_hours:
            current_hour = current_time.hour
            if not (market_hours['start'] <= current_hour <= market_hours['end']):
                logger.debug(f"Outside market hours: {current_hour}")
                return False
        
        # Check signal freshness
        signal_age = (current_time - signal.timestamp).total_seconds() / 60  # minutes
        max_signal_age = self.config.get('max_signal_age_minutes', 15)
        
        if signal_age > max_signal_age:
            logger.debug(f"Signal too old: {signal_age:.1f} minutes")
            return False
        
        # Check for recent trades (avoid overtrading)
        min_time_between_trades = self.config.get('min_time_between_trades_minutes', 30)
        recent_trades = [
            trade for trade in self.trade_history 
            if (current_time - trade['timestamp']).total_seconds() / 60 < min_time_between_trades
        ]
        
        if len(recent_trades) >= self.config.get('max_trades_per_period', 3):
            logger.debug("Too many recent trades")
            return False
        
        return True
    
    async def _calculate_position_size(self, signal: TradeSignal,
                                      account_info: AccountInfo,
                                      current_positions: List[Position]) -> float:
        """Calculate optimal position size"""
        
        # Base risk amount (1% of equity)
        base_risk_amount = account_info.equity * (self.base_risk_per_trade / 100)
        
        # Adjust for confidence
        if self.confidence_adjustment:
            confidence_multiplier = signal.confidence
            base_risk_amount *= confidence_multiplier
        
        # Adjust for volatility
        if self.volatility_adjustment:
            # Higher volatility = smaller position
            volatility_multiplier = max(0.5, 1.0 - (signal.volatility * 10))
            base_risk_amount *= volatility_multiplier
        
        # Calculate position size based on stop loss distance
        if signal.stop_loss and signal.entry_price:
            stop_loss_distance = abs(signal.entry_price - signal.stop_loss)
            if stop_loss_distance > 0:
                # Risk amount / (stop loss distance * contract size)
                position_size = base_risk_amount / (stop_loss_distance * 100000)  # Assuming 100k contract size
            else:
                position_size = 0.0
        else:
            # Default position size based on account size
            position_size = base_risk_amount / (signal.current_price * 1000)
        
        # Apply maximum position size limit
        max_size_by_account = account_info.equity * (self.max_position_size / 100) / signal.current_price
        position_size = min(position_size, max_size_by_account)
        
        # Apply signal's max position size
        position_size = min(position_size, signal.max_position_size)
        
        # Round to valid lot size (typically 0.01 for forex)
        position_size = round(position_size, 2)
        
        # Ensure minimum position size
        min_position_size = self.config.get('min_position_size', 0.01)
        if position_size < min_position_size:
            position_size = 0.0
        
        logger.debug(f"Calculated position size: {position_size} lots")
        return position_size
    
    async def _create_trade_setup(self, signal: TradeSignal, 
                                position_size: float,
                                account_info: AccountInfo) -> TradeSetup:
        """Create complete trade setup"""
        
        # Determine order type
        if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            from ..models.trading_models import OrderType
            order_type = OrderType.BUY
            entry_price = signal.entry_price or signal.current_price * 1.001  # Small premium
        else:
            order_type = OrderType.SELL
            entry_price = signal.entry_price or signal.current_price * 0.999  # Small discount
        
        # Calculate stop loss and take profit
        stop_loss = signal.stop_loss
        take_profit = signal.take_profit
        
        if not stop_loss:
            # Default stop loss (2% for long, 2% for short)
            if order_type == OrderType.BUY:
                stop_loss = entry_price * 0.98
            else:
                stop_loss = entry_price * 1.02
        
        if not take_profit:
            # Default take profit (3:1 risk reward)
            if order_type == OrderType.BUY:
                risk_distance = entry_price - stop_loss
                take_profit = entry_price + (risk_distance * 3)
            else:
                risk_distance = stop_loss - entry_price
                take_profit = entry_price - (risk_distance * 3)
        
        # Calculate risk metrics
        if order_type == OrderType.BUY:
            risk_per_unit = entry_price - stop_loss
            reward_per_unit = take_profit - entry_price
        else:
            risk_per_unit = stop_loss - entry_price
            reward_per_unit = entry_price - take_profit
        
        risk_amount = position_size * risk_per_unit * 100000  # Contract size
        expected_return = position_size * reward_per_unit * 100000
        risk_reward_ratio = reward_per_unit / risk_per_unit if risk_per_unit > 0 else 0
        
        trade_setup = TradeSetup(
            signal=signal,
            position_size=position_size,
            order_type=order_type,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_amount=abs(risk_amount),
            expected_return=expected_return,
            risk_reward_ratio=risk_reward_ratio
        )
        
        return trade_setup
    
    def _validate_risk_metrics(self, trade_setup: TradeSetup) -> bool:
        """Validate risk metrics of trade setup"""
        
        # Check risk-reward ratio
        if trade_setup.risk_reward_ratio < 1.5:
            logger.debug(f"Poor risk-reward ratio: {trade_setup.risk_reward_ratio}")
            return False
        
        # Check risk amount
        max_risk_amount = self.max_historical_equity * (self.base_risk_per_trade / 100)
        if trade_setup.risk_amount > max_risk_amount:
            logger.debug(f"Risk amount too high: {trade_setup.risk_amount}")
            return False
        
        return True
    
    def _validate_position_size(self, trade_setup: TradeSetup, account_info: AccountInfo) -> bool:
        """Validate position size"""
        
        # Check minimum size
        if trade_setup.position_size < 0.01:
            return False
        
        # Check maximum size
        max_size = account_info.equity * 0.1 / trade_setup.signal.current_price  # 10% max
        if trade_setup.position_size > max_size:
            return False
        
        return True
    
    def _validate_timing(self, trade_setup: TradeSetup) -> bool:
        """Validate trade timing"""
        # Additional timing checks can be added here
        return True
    
    async def update_daily_pnl(self, current_positions: List[Position]):
        """Update daily P&L tracking"""
        try:
            total_unrealized = sum(pos.unrealized_pnl for pos in current_positions)
            
            # This would typically include realized P&L from closed trades today
            # For now, just using unrealized P&L
            self.daily_pnl = total_unrealized
            
        except Exception as e:
            logger.error(f"Error updating daily P&L: {e}")