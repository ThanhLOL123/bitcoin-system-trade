import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from ..models.trading_models import TradeSignal, TradeSetup, SignalType
from ..risk_management.risk_manager import RiskManager
from ..mt5_connector import MT5Connector
from ..portfolio.portfolio_manager import PortfolioManager

logger = logging.getLogger(__name__)

class StrategyEngine:
    """Main trading strategy engine"""
    
    def __init__(self, config: Dict, mt5_connector: MT5Connector, portfolio_manager: PortfolioManager):
        self.config = config
        self.mt5_connector = mt5_connector
        self.portfolio_manager = portfolio_manager
        self.risk_manager = RiskManager(config['risk'])
        
        # Strategy state
        self.active_signals = {}
        self.strategy_performance = {}
        self.is_running = False
        
        # Strategy parameters
        self.signal_refresh_interval = config.get('signal_refresh_seconds', 300)  # 5 minutes
        self.max_concurrent_positions = config.get('max_concurrent_positions', 5)
        self.strategies_enabled = config.get('strategies_enabled', ['ml_ensemble'])
        
    async def start(self):
        """Start the strategy engine"""
        try:
            self.is_running = True
            logger.info("Strategy engine started")
            
            # Start main strategy loop
            await self._run_strategy_loop()
            
        except Exception as e:
            logger.error(f"Strategy engine error: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the strategy engine"""
        self.is_running = False
        
        # Close all positions if configured
        if self.config.get('close_positions_on_stop', False):
            await self._close_all_positions()
        
        logger.info("Strategy engine stopped")
    
    async def _run_strategy_loop(self):
        """Main strategy execution loop"""
        while self.is_running:
            try:
                # Get current market state
                await self._update_market_state()
                
                # Generate signals from enabled strategies
                signals = await self._generate_signals()
                
                # Process signals
                for signal in signals:
                    await self._process_signal(signal)
                
                # Manage existing positions
                await self._manage_positions()
                
                # Update portfolio metrics
                await self.portfolio_manager.update_metrics(self.mt5_connector.account_info(), self.mt5_connector.get_positions())
                
                # Wait for next iteration
                await asyncio.sleep(self.signal_refresh_interval)
                
            except Exception as e:
                logger.error(f"Strategy loop error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def _update_market_state(self):
        """Update current market state"""
        try:
            # Update symbol info
            symbol_info = self.mt5_connector.symbol_info(self.config['mt5']['symbol'])
            
            # Update account info
            account_info = self.mt5_connector.account_info()
            
            # Update positions
            positions = self.mt5_connector.get_positions()
            
            # Update portfolio manager
            await self.portfolio_manager.update_state(account_info, positions, symbol_info)
            
            # Update risk manager
            await self.risk_manager.update_daily_pnl(positions)
            
        except Exception as e:
            logger.error(f"Error updating market state: {e}")
    
    async def _generate_signals(self) -> List[TradeSignal]:
        """Generate trading signals from all enabled strategies"""
        all_signals = []
        
        for strategy_name in self.strategies_enabled:
            try:
                if strategy_name == 'ml_ensemble':
                    signals = await self._generate_ml_signals()
                elif strategy_name == 'technical_analysis':
                    signals = await self._generate_technical_signals()
                elif strategy_name == 'mean_reversion':
                    signals = await self._generate_mean_reversion_signals()
                else:
                    logger.warning(f"Unknown strategy: {strategy_name}")
                    continue
                
                # Add strategy name to signals
                for signal in signals:
                    signal.model_name = f"{strategy_name}_{signal.model_name}"
                
                all_signals.extend(signals)
                
            except Exception as e:
                logger.error(f"Error generating {strategy_name} signals: {e}")
        
        return all_signals
    
    async def _generate_ml_signals(self) -> List[TradeSignal]:
        """Generate signals from ML models"""
        signals = []
        
        try:
            # Get predictions from API (this would be replaced with direct service call)
            # For now, creating mock signal
            signal = await self._create_ml_signal_from_prediction()
            if signal:
                signals.append(signal)
            
        except Exception as e:
            logger.error(f"Error generating ML signals: {e}")
        
        return signals
    
    async def _create_ml_signal_from_prediction(self) -> Optional[TradeSignal]:
        """Create trading signal from ML prediction"""
        try:
            # Mock prediction data - replace with actual API call
            current_price = 45000.0
            predicted_price = current_price * (1 + np.random.randn() * 0.02)
            confidence = 0.75 + np.random.random() * 0.2
            
            price_change_percent = (predicted_price - current_price) / current_price * 100
            
            # Determine signal type based on prediction
            if price_change_percent > 1.0 and confidence > 0.8:
                signal_type = SignalType.STRONG_BUY
            elif price_change_percent > 0.5:
                signal_type = SignalType.BUY
            elif price_change_percent < -1.0 and confidence > 0.8:
                signal_type = SignalType.STRONG_SELL
            elif price_change_percent < -0.5:
                signal_type = SignalType.SELL
            else:
                signal_type = SignalType.HOLD
            
            if signal_type == SignalType.HOLD:
                return None
            
            # Calculate entry, stop loss, and take profit
            if signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                entry_price = current_price * 1.001
                stop_loss = entry_price * 0.98
                take_profit = entry_price * 1.06
            else:
                entry_price = current_price * 0.999
                stop_loss = entry_price * 1.02
                take_profit = entry_price * 0.94
            
            # Calculate risk-reward ratio
            if signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                risk = entry_price - stop_loss
                reward = take_profit - entry_price
            else:
                risk = stop_loss - entry_price
                reward = entry_price - take_profit
            
            risk_reward_ratio = reward / risk if risk > 0 else 0
            
            if risk_reward_ratio < 1.5:
                return None
            
            signal = TradeSignal(
                timestamp=datetime.now(),
                symbol=self.config['mt5']['symbol'],
                signal_type=signal_type,
                confidence=confidence,
                predicted_price=predicted_price,
                current_price=current_price,
                price_change_percent=price_change_percent,
                horizon="1h",
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=risk_reward_ratio,
                model_name="ml_ensemble",
                model_confidence=confidence,
                volatility=0.02,
                max_position_size=1.0
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error creating ML signal: {e}")
            return None
    
    async def _generate_technical_signals(self) -> List[TradeSignal]:
        """Generate signals from technical analysis"""
        signals = []
        
        try:
            # Get historical data
            df = self.mt5_connector.get_historical_data(self.config['mt5']['symbol'], self.mt5_connector.mt5.TIMEFRAME_H1, 200)
            
            if df.empty:
                return signals
            
            # Calculate technical indicators
            df = self._add_technical_indicators(df)
            
            # Generate signal based on technical conditions
            signal = self._evaluate_technical_conditions(df)
            
            if signal:
                signals.append(signal)
                
        except Exception as e:
            logger.error(f"Error generating technical signals: {e}")
        
        return signals
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to dataframe"""
        try:
            # Simple moving averages
            df['sma_20'] = df['close'].rolling(20).mean()
            df['sma_50'] = df['close'].rolling(50).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            
            # Bollinger Bands
            sma_20 = df['close'].rolling(20).mean()
            std_20 = df['close'].rolling(20).std()
            df['bb_upper'] = sma_20 + (std_20 * 2)
            df['bb_lower'] = sma_20 - (std_20 * 2)
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            return df
    
    def _evaluate_technical_conditions(self, df: pd.DataFrame) -> Optional[TradeSignal]:
        """Evaluate technical conditions for signal generation"""
        try:
            if len(df) < 50:
                return None
            
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            current_price = latest['close']
            
            # Bullish conditions
            bullish_score = 0
            if latest['close'] > latest['sma_20']:
                bullish_score += 1
            if latest['sma_20'] > latest['sma_50']:
                bullish_score += 1
            if latest['rsi'] < 70 and latest['rsi'] > 50:
                bullish_score += 1
            if latest['macd'] > latest['macd_signal']:
                bullish_score += 1
            if latest['close'] > prev['bb_lower'] and latest['close'] < latest['bb_upper']:
                bullish_score += 1
            
            # Bearish conditions
            bearish_score = 0
            if latest['close'] < latest['sma_20']:
                bearish_score += 1
            if latest['sma_20'] < latest['sma_50']:
                bearish_score += 1
            if latest['rsi'] > 30 and latest['rsi'] < 50:
                bearish_score += 1
            if latest['macd'] < latest['macd_signal']:
                bearish_score += 1
            if latest['close'] < prev['bb_upper'] and latest['close'] > latest['bb_lower']:
                bearish_score += 1
            
            # Generate signal
            signal_type = None
            confidence = 0.0
            
            if bullish_score >= 4:
                signal_type = SignalType.BUY
                confidence = bullish_score / 5.0
            elif bearish_score >= 4:
                signal_type = SignalType.SELL
                confidence = bearish_score / 5.0
            
            if not signal_type:
                return None
            
            # Calculate entry, stop, and target levels
            if signal_type == SignalType.BUY:
                entry_price = current_price * 1.001
                stop_loss = min(latest['sma_20'], latest['bb_lower']) * 0.995
                take_profit = latest['bb_upper'] * 1.005
            else:
                entry_price = current_price * 0.999
                stop_loss = max(latest['sma_20'], latest['bb_upper']) * 1.005
                take_profit = latest['bb_lower'] * 0.995
            
            # Calculate risk-reward
            if signal_type == SignalType.BUY:
                risk = entry_price - stop_loss
                reward = take_profit - entry_price
            else:
                risk = stop_loss - entry_price
                reward = entry_price - take_profit
            
            risk_reward_ratio = reward / risk if risk > 0 else 0
            
            if risk_reward_ratio < 1.5:
                return None
            
            signal = TradeSignal(
                timestamp=datetime.now(),
                symbol=self.config['mt5']['symbol'],
                signal_type=signal_type,
                confidence=confidence,
                predicted_price=current_price,
                current_price=current_price,
                price_change_percent=0.0,
                horizon="1h",
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=risk_reward_ratio,
                model_name="technical_analysis",
                model_confidence=confidence,
                volatility=0.02,
                max_position_size=1.0
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error evaluating technical conditions: {e}")
            return None
    
    async def _generate_mean_reversion_signals(self) -> List[TradeSignal]:
        """Generate mean reversion signals"""
        # Implementation would go here
        return []
    
    async def _process_signal(self, signal: TradeSignal):
        """Process a trading signal"""
        try:
            logger.info(f"Processing signal: {signal.signal_type.value} {signal.symbol}")
            
            # Get current account and position info
            account_info = self.mt5_connector.account_info()
            current_positions = self.mt5_connector.get_positions()
            
            # Check if we already have a position in this direction
            if self._has_conflicting_position(signal, current_positions):
                logger.debug(f"Skipping signal due to conflicting position")
                return
            
            # Check maximum concurrent positions
            if len(current_positions) >= self.max_concurrent_positions:
                logger.debug(f"Max concurrent positions reached: {len(current_positions)}")
                return
            
            # Validate signal through risk manager
            trade_setup = await self.risk_manager.validate_trade_signal(
                signal, account_info, current_positions
            )
            
            # Execute trade
            await self._execute_trade(trade_setup)
            
        except Exception as e:
            logger.error(f"Error processing signal: {e}")
    
    def _has_conflicting_position(self, signal: TradeSignal, positions: List[Position]) -> bool:
        """Check if signal conflicts with existing positions"""
        
        for position in positions:
            if position.symbol != signal.symbol:
                continue
            
            # Check for conflicting directions
            if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                if position.side.value == "LONG":
                    return True  # Already long
            elif signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                if position.side.value == "SHORT":
                    return True  # Already short
        
        return False
    
    async def _execute_trade(self, trade_setup: TradeSetup):
        """Execute validated trade"""
        try:
            from ..models.trading_models import OrderRequest
            
            # Create order request
            order_request = OrderRequest(
                symbol=trade_setup.signal.symbol,
                order_type=trade_setup.order_type,
                volume=trade_setup.position_size,
                price=trade_setup.entry_price,
                stop_loss=trade_setup.stop_loss,
                take_profit=trade_setup.take_profit,
                magic_number=self.config['mt5']['magic_number'],
                comment=f"Strategy: {trade_setup.signal.model_name}"
            )
            
            # Execute order
            result = self.mt5_connector.place_order(order_request.symbol, order_request.order_type.value, order_request.volume, order_request.price, order_request.stop_loss, order_request.take_profit)
            
            if result and result.retcode == self.mt5_connector.mt5.TRADE_RETCODE_DONE:
                logger.info(f"Trade executed successfully: {result.order}")
                
                # Record trade in portfolio manager
                await self.portfolio_manager.record_trade(trade_setup, result)
                
            else:
                logger.error(f"Trade execution failed: {result.comment}")
                
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
    
    async def _manage_positions(self):
        """Manage existing positions"""
        try:
            positions = self.mt5_connector.get_positions()
            
            for position in positions:
                await self._manage_individual_position(position)
                
        except Exception as e:
            logger.error(f"Error managing positions: {e}")
    
    async def _manage_individual_position(self, position: Position):
        """Manage individual position"""
        try:
            # Check for trailing stop updates
            await self._update_trailing_stops(position)
            
            # Check for time-based exits
            await self._check_time_based_exits(position)
            
            # Check for emergency exits
            await self._check_emergency_exits(position)
            
        except Exception as e:
            logger.error(f"Error managing position {position.ticket}: {e}")
    
    async def _update_trailing_stops(self, position: Position):
        """Update trailing stops for position"""
        # Implementation would depend on specific trailing stop logic
        pass
    
    async def _check_time_based_exits(self, position: Position):
        """Check for time-based position exits"""
        max_position_time = self.config.get('max_position_time_hours', 24)
        
        position_age = datetime.now() - position.open_time
        if position_age.total_seconds() / 3600 > max_position_time:
            logger.info(f"Closing position {position.ticket} due to time limit")
            # await self.mt5_connector.close_position(str(position.ticket))
    
    async def _check_emergency_exits(self, position: Position):
        """Check for emergency position exits"""
        # Implement emergency exit conditions
        max_loss_percent = self.config.get('emergency_exit_loss_percent', 5.0)
        
        if position.profit < 0:
            loss_percent = abs(position.profit) / (position.volume * position.open_price) * 100
            if loss_percent > max_loss_percent:
                logger.warning(f"Emergency exit for position {position.ticket}")
                # await self.mt5_connector.close_position(str(position.ticket))
