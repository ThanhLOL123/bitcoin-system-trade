import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from src.trading_bot.strategy.strategy_engine import StrategyEngine
from src.trading_bot.models.trading_models import TradeSignal, SignalType, OrderType, TradeSetup, Position, AccountInfo, MarketInfo
from datetime import datetime, timedelta
import pandas as pd

@pytest.fixture
def mock_mt5_connector():
    mock = AsyncMock()
    mock.account_info.return_value = AccountInfo(login=123, balance=10000, equity=10000, margin=100, free_margin=9900, margin_level=1000, currency="USD")
    mock.get_positions.return_value = []
    mock.symbol_info.return_value = MarketInfo(symbol="BTCUSD", bid=50000, ask=50010, spread=10, point=1, digits=5, tick_value=1, tick_size=1, contract_size=1, min_lot=0.01, max_lot=10, lot_step=0.01)
    mock.get_historical_data.return_value = pd.DataFrame({'close': [49000, 49500, 50000, 50500, 51000]})
    mock.place_order.return_value = MagicMock(retcode=0, order=12345, price=50000, volume=0.01, comment="test")
    return mock

@pytest.fixture
def mock_portfolio_manager():
    mock = AsyncMock()
    mock.update_state = AsyncMock()
    mock.update_metrics = AsyncMock()
    mock.record_trade = AsyncMock()
    return mock

@pytest.fixture
def strategy_config():
    return {
        'mt5': {'symbol': 'BTCUSD', 'magic_number': 123456},
        'risk': {
            'max_daily_loss_percent': 2.0,
            'max_position_size_percent': 5.0,
            'max_total_exposure_percent': 20.0,
            'max_drawdown_percent': 10.0,
            'base_risk_per_trade_percent': 1.0,
            'volatility_adjustment': True,
            'confidence_adjustment': True,
            'min_signal_confidence': 0.6,
            'min_model_confidence': 0.7,
            'min_price_change_percent': 0.5,
            'min_margin_level': 200.0
        },
        'portfolio': {},
        'strategies_enabled': ['ml_ensemble', 'technical_analysis'],
        'signal_refresh_seconds': 1,
        'close_positions_on_stop': False,
        'max_position_time_hours': 24,
        'emergency_exit_loss_percent': 5.0
    }

@pytest.fixture
def strategy_engine(strategy_config, mock_mt5_connector, mock_portfolio_manager):
    return StrategyEngine(strategy_config, mock_mt5_connector, mock_portfolio_manager)

@pytest.mark.asyncio
async def test_start_and_stop(strategy_engine):
    with patch.object(strategy_engine, '_run_strategy_loop', new=AsyncMock()):
        await strategy_engine.start()
        assert strategy_engine.is_running is True
        await strategy_engine.stop()
        assert strategy_engine.is_running is False

@pytest.mark.asyncio
async def test_update_market_state(strategy_engine):
    await strategy_engine._update_market_state()
    strategy_engine.mt5_connector.account_info.assert_called_once()
    strategy_engine.mt5_connector.get_positions.assert_called_once()
    strategy_engine.mock_portfolio_manager.update_state.assert_called_once()
    strategy_engine.risk_manager.update_daily_pnl.assert_called_once()

@pytest.mark.asyncio
async def test_generate_ml_signals(strategy_engine):
    with patch.object(strategy_engine, '_create_ml_signal_from_prediction', new=AsyncMock(return_value=TradeSignal(timestamp=datetime.now(), symbol="BTCUSD", signal_type=SignalType.BUY, confidence=0.8, predicted_price=51000, current_price=50000, price_change_percent=2.0, horizon="1h", model_name="test", model_confidence=0.8, volatility=0.02, max_position_size=1.0))):
        signals = await strategy_engine._generate_ml_signals()
        assert len(signals) == 1
        assert signals[0].signal_type == SignalType.BUY

@pytest.mark.asyncio
async def test_process_signal_no_conflicting_position(strategy_engine):
    signal = TradeSignal(timestamp=datetime.now(), symbol="BTCUSD", signal_type=SignalType.BUY, confidence=0.8, predicted_price=51000, current_price=50000, price_change_percent=2.0, horizon="1h", model_name="test", model_confidence=0.8, volatility=0.02, max_position_size=1.0)
    
    with patch.object(strategy_engine.risk_manager, 'validate_trade_signal', new=AsyncMock(return_value=TradeSetup(signal=signal, position_size=0.01, order_type=OrderType.BUY, entry_price=50000, stop_loss=49500, take_profit=50500, risk_amount=500, expected_return=500, risk_reward_ratio=1))):
        with patch.object(strategy_engine, '_execute_trade', new=AsyncMock()) as mock_execute_trade:
            await strategy_engine._process_signal(signal)
            mock_execute_trade.assert_called_once()

@pytest.mark.asyncio
async def test_execute_trade(strategy_engine):
    signal = TradeSignal(timestamp=datetime.now(), symbol="BTCUSD", signal_type=SignalType.BUY, confidence=0.8, predicted_price=51000, current_price=50000, price_change_percent=2.0, horizon="1h", model_name="test", model_confidence=0.8, volatility=0.02, max_position_size=1.0)
    trade_setup = TradeSetup(signal=signal, position_size=0.01, order_type=OrderType.BUY, entry_price=50000, stop_loss=49500, take_profit=50500, risk_amount=500, expected_return=500, risk_reward_ratio=1)
    
    await strategy_engine._execute_trade(trade_setup)
    strategy_engine.mt5_connector.place_order.assert_called_once()
    strategy_engine.mock_portfolio_manager.record_trade.assert_called_once()
