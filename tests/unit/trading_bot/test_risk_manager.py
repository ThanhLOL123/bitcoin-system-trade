import pytest
from unittest.mock import MagicMock
from src.trading_bot.risk_management.risk_manager import RiskManager

@pytest.fixture
def mock_account_info():
    mock = MagicMock()
    mock.balance = 10000.0
    mock.equity = 10000.0
    mock.free_margin = 9000.0
    mock.margin_level = 500.0
    return mock

@pytest.fixture
def mock_symbol_info():
    mock = MagicMock()
    mock.point = 0.00001
    mock.trade_tick_size = 0.00001
    mock.trade_tick_value = 1.0
    mock.volume_min = 0.01
    mock.volume_max = 100.0
    mock.volume_step = 0.01
    return mock

@pytest.fixture
def risk_manager(mock_account_info, mock_symbol_info):
    return RiskManager(mock_account_info, mock_symbol_info)

def test_calculate_position_size_positive_sl(risk_manager):
    risk_per_trade_percent = 1.0
    stop_loss_price = 1.23000
    entry_price = 1.23500
    
    # 500 pips difference
    # Risk amount = 10000 * 0.01 = 100
    # Lots = 100 / (500 * 0.00001 * 1.0) = 100 / 0.005 = 20000 (too high, will be capped)
    
    lots = risk_manager.calculate_position_size(risk_per_trade_percent, stop_loss_price, entry_price)
    assert lots == risk_manager.symbol_info.volume_max # Should be capped at max_volume

def test_calculate_position_size_zero_sl(risk_manager):
    risk_per_trade_percent = 1.0
    stop_loss_price = 1.23450
    entry_price = 1.23450
    
    lots = risk_manager.calculate_position_size(risk_per_trade_percent, stop_loss_price, entry_price)
    assert lots == 0.0

def test_check_max_drawdown_below_limit(risk_manager):
    current_equity = 9500.0
    peak_equity = 10000.0
    max_drawdown_percent = 10.0
    assert risk_manager.check_max_drawdown(current_equity, peak_equity, max_drawdown_percent) is False

def test_check_max_drawdown_above_limit(risk_manager):
    current_equity = 8500.0
    peak_equity = 10000.0
    max_drawdown_percent = 10.0
    assert risk_manager.check_max_drawdown(current_equity, peak_equity, max_drawdown_percent) is True

def test_check_max_drawdown_zero_peak_equity(risk_manager):
    current_equity = 100.0
    peak_equity = 0.0
    max_drawdown_percent = 10.0
    assert risk_manager.check_max_drawdown(current_equity, peak_equity, max_drawdown_percent) is False
