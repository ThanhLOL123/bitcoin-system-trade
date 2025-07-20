import pytest
from src.trading_bot.portfolio.portfolio_manager import PortfolioManager

@pytest.fixture
def portfolio_manager():
    return PortfolioManager()

def test_update_equity(portfolio_manager):
    portfolio_manager.update_equity(10000.0)
    assert portfolio_manager.equity_history == [10000.0]

def test_update_pnl(portfolio_manager):
    portfolio_manager.update_pnl(100.0)
    assert portfolio_manager.pnl_history == [100.0]

def test_record_trade(portfolio_manager):
    trade_details = {"symbol": "BTCUSD", "type": "BUY"}
    portfolio_manager.record_trade(trade_details)
    assert portfolio_manager.trade_history == [trade_details]

def test_get_performance_summary_empty(portfolio_manager):
    summary = portfolio_manager.get_performance_summary()
    assert summary["total_trades"] == 0
    assert summary["last_equity"] == 0
    assert summary["last_pnl"] == 0

def test_get_performance_summary_with_data(portfolio_manager):
    portfolio_manager.update_equity(10000.0)
    portfolio_manager.update_pnl(100.0)
    portfolio_manager.record_trade({"symbol": "BTCUSD", "type": "BUY"})
    summary = portfolio_manager.get_performance_summary()
    assert summary["total_trades"] == 1
    assert summary["last_equity"] == 10000.0
    assert summary["last_pnl"] == 100.0
