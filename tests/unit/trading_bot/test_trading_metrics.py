import pytest
from unittest.mock import patch
from src.trading_bot.monitoring.trading_metrics import TradingMetrics, TRADES_TOTAL, PNL_GAUGE, EQUITY_GAUGE, DRAWDOWN_GAUGE, ORDER_LATENCY_HISTOGRAM

@pytest.fixture(autouse=True)
def reset_metrics():
    # Reset metrics before each test to ensure isolation
    TRADES_TOTAL._metrics.clear()
    PNL_GAUGE._value = 0.0
    EQUITY_GAUGE._value = 0.0
    DRAWDOWN_GAUGE._value = 0.0
    ORDER_LATENCY_HISTOGRAM._sum = 0.0
    ORDER_LATENCY_HISTOGRAM._count = 0

def test_record_trade():
    metrics = TradingMetrics()
    metrics.record_trade("BUY", "BTCUSD")
    assert TRADES_TOTAL._metrics[(('BUY', 'BTCUSD'),)] == 1.0

def test_set_pnl():
    metrics = TradingMetrics()
    metrics.set_pnl(100.50)
    assert PNL_GAUGE._value == 100.50

def test_set_equity():
    metrics = TradingMetrics()
    metrics.set_equity(50000.0)
    assert EQUITY_GAUGE._value == 50000.0

def test_set_drawdown():
    metrics = TradingMetrics()
    metrics.set_drawdown(5.25)
    assert DRAWDOWN_GAUGE._value == 5.25

def test_observe_order_latency():
    metrics = TradingMetrics()
    metrics.observe_order_latency(0.123)
    assert ORDER_LATENCY_HISTOGRAM._sum == 0.123
    assert ORDER_LATENCY_HISTOGRAM._count == 1
