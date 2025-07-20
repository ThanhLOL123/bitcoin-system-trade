import pytest
import asyncio
import httpx
import time
import os
from unittest.mock import AsyncMock, patch, MagicMock
from src.trading_bot.trading_bot import TradingBot
from src.trading_bot.mt5_connector import MT5Connector
from src.ml_pipeline.serving.api import router as ml_api_router
from src.feature_engineering.feature_store import FeatureStore
import pandas as pd
from datetime import datetime

# Configuration for the trading bot (adjust as needed)
TRADING_BOT_CONFIG = {
    'mt5': {
        'login': 12345678,
        'password': 'your-mt5-password',
        'server': 'YourBroker-Server',
        'symbol': 'BTCUSD',
        'magic_number': 123456,
        'mt5_rpc_host': 'localhost', # Assuming RPC server is accessible directly
        'mt5_rpc_port': 18812
    },
    'risk': {
        'max_daily_loss_percent': 10.0,
        'max_position_size_percent': 10.0,
        'max_total_exposure_percent': 50.0,
        'max_drawdown_percent': 20.0,
        'base_risk_per_trade_percent': 1.0,
        'volatility_adjustment': False,
        'confidence_adjustment': False,
        'min_signal_confidence': 0.1,
        'min_model_confidence': 0.1,
        'min_price_change_percent': 0.0,
        'min_margin_level': 10.0
    },
    'portfolio': {
        'initial_balance': 10000.0,
        'max_concurrent_positions': 1,
        'strategy_allocation': {}
    },
    'strategies_enabled': ['ml_ensemble'],
    'signal_refresh_seconds': 1,
    'close_positions_on_stop': False,
    'max_position_time_hours': 1,
    'emergency_exit_loss_percent': 5.0
}

# Mock ML API URL (assuming it's running locally for E2E test)
ML_API_URL = "http://localhost:8000/ml"

@pytest.fixture(scope="module", autouse=True)
async def setup_e2e_environment():
    # This fixture assumes Docker Compose services are already running
    # and accessible. In a real CI/CD, this would involve starting
    # Docker Compose or deploying to a test Kubernetes cluster.
    print("\nSetting up E2E environment...")
    # Give services some time to start up
    await asyncio.sleep(10) 
    print("E2E environment setup complete.")
    yield
    print("Tearing down E2E environment (no action taken, assumes external teardown).")

@pytest.mark.asyncio
async def test_full_trading_system_flow(setup_e2e_environment):
    print("\nStarting full trading system E2E test...")

    # 1. Simulate data collection (optional, as data collector runs independently)
    # For this E2E test, we'll assume data is being collected and processed
    # by the data_collector and feature_engineering services.
    
    # 2. Simulate ML model training (optional, as trainer runs independently)
    # For this E2E test, we'll assume a model has been trained and registered.
    # We need to mock the MLflow model loading in the API for this test.
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1]) # Simulate a BUY signal

    with patch('mlflow.pyfunc.load_model', return_value=mock_model):
        with patch('mlflow.tracking.MlflowClient.get_latest_versions', return_value=[MagicMock(version=1)]):
            # Manually trigger ML API startup event to load the mocked model
            await ml_api_router.startup()

            # 3. Simulate feature store data for ML prediction
            feature_store = FeatureStore()
            dummy_features = pd.DataFrame({
                'price_usd': [50000],
                'volume_24h': [100000],
                'feature1': [0.5],
                'feature2': [0.7],
                'target': [0] # Dummy target
            })
            feature_store.save_features(dummy_features, datetime.now().strftime('%Y-%m-%d'))

            # 4. Initialize and run the Trading Bot
            trading_bot = TradingBot(
                login=TRADING_BOT_CONFIG['mt5']['login'],
                password=TRADING_BOT_CONFIG['mt5']['password'],
                server=TRADING_BOT_CONFIG['mt5']['server'],
                mt5_rpc_host=TRADING_BOT_CONFIG['mt5']['mt5_rpc_host'],
                mt5_rpc_port=TRADING_BOT_CONFIG['mt5']['mt5_rpc_port'],
                symbol=TRADING_BOT_CONFIG['mt5']['symbol'],
                lot=0.01,
                max_positions=TRADING_BOT_CONFIG['portfolio']['max_concurrent_positions'],
                risk_per_trade_percent=TRADING_BOT_CONFIG['risk']['base_risk_per_trade_percent'],
                max_drawdown_percent=TRADING_BOT_CONFIG['risk']['max_drawdown_percent'],
                ml_api_url=ML_API_URL
            )

            # Mock MT5Connector methods to prevent actual trading
            with patch.object(trading_bot.mt5_connector, 'connect', new=AsyncMock(return_value=True)):
                with patch.object(trading_bot.mt5_connector, 'account_info', new=MagicMock(return_value=MagicMock(equity=10000, profit=0))):
                    with patch.object(trading_bot.mt5_connector, 'symbol_info', new=MagicMock(return_value=MagicMock(point=1, trade_tick_size=1, trade_tick_value=1, volume_min=0.01, volume_max=100.0, volume_step=0.01))):
                        with patch.object(trading_bot.mt5_connector, 'symbol_info_tick', new=MagicMock(last=50000)):
                            with patch.object(trading_bot.mt5_connector, 'get_positions', new=MagicMock(return_value=[])):
                                with patch.object(trading_bot.mt5_connector, 'place_order', new=MagicMock(return_value=MagicMock(retcode=0, order=12345, comment="test"))):
                                    with patch.object(trading_bot.mt5_connector, 'disconnect', new=MagicMock()):
                                        # Run the bot for a short period to allow a trade to be simulated
                                        # We need to mock asyncio.sleep to prevent actual delays
                                        with patch('asyncio.sleep', new=AsyncMock()) as mock_sleep:
                                            # Simulate one loop iteration
                                            trading_bot.is_running = True
                                            await trading_bot.run() # This will run one iteration and then break
                                            trading_bot.is_running = False # Manually stop after one iteration

                                            # 5. Verify a trade was attempted
                                            trading_bot.mt5_connector.place_order.assert_called_once()
                                            args, kwargs = trading_bot.mt5_connector.place_order.call_args
                                            assert args[1] == "BUY" # Expecting a BUY order from mock ML prediction
                                            assert args[2] > 0 # Position size should be calculated

                                            print("Full trading system E2E test completed successfully.")