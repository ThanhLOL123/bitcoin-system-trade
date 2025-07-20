from prometheus_client import Counter, Gauge, Histogram

# Define Prometheus metrics
TRADES_TOTAL = Counter('trades_total', 'Total number of trades executed', ['type', 'symbol'])
PNL_GAUGE = Gauge('current_pnl', 'Current PnL of the trading bot')
EQUITY_GAUGE = Gauge('account_equity', 'Current account equity')
DRAWDOWN_GAUGE = Gauge('current_drawdown', 'Current drawdown percentage')
ORDER_LATENCY_HISTOGRAM = Histogram('order_latency_seconds', 'Latency of order execution')

class TradingMetrics:
    """Helper class to update trading-related Prometheus metrics"""
    
    def record_trade(self, trade_type: str, symbol: str):
        TRADES_TOTAL.labels(type=trade_type, symbol=symbol).inc()

    def set_pnl(self, pnl: float):
        PNL_GAUGE.set(pnl)

    def set_equity(self, equity: float):
        EQUITY_GAUGE.set(equity)

    def set_drawdown(self, drawdown: float):
        DRAWDOWN_GAUGE.set(drawdown)

    def observe_order_latency(self, latency: float):
        ORDER_LATENCY_HISTOGRAM.observe(latency)
