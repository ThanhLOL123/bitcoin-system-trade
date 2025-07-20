TRADING_CONFIG = {
    'mt5': {
        'login': 12345678,
        'password': 'your-mt5-password',
        'server': 'YourBroker-Server',
        'symbol': 'BTCUSD',
        'magic_number': 123456,
        'mt5_rpc_host': 'mt5-server',
        'mt5_rpc_port': 18812
    },
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
    'portfolio': {
        'initial_balance': 10000.0,
        'max_concurrent_positions': 5,
        'strategy_allocation': {
            'ml_ensemble': 0.6,
            'technical_analysis': 0.4
        }
    },
    'strategies_enabled': [
        'ml_ensemble',
        'technical_analysis'
    ],
    'signal_refresh_seconds': 300,
    'close_positions_on_stop': True,
    'max_position_time_hours': 48,
    'emergency_exit_loss_percent': 5.0
}
