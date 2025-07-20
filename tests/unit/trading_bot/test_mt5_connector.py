import pytest
from unittest.mock import MagicMock, patch
from src.trading_bot.mt5_connector import MT5Connector
from mt5linux import MetaTrader5

@pytest.fixture
def mock_mt5linux():
    with patch('mt5linux.MetaTrader5') as MockMT5Linux:
        mock_instance = MockMT5Linux.return_value
        mock_instance.initialize.return_value = True
        mock_instance.login.return_value = True
        mock_instance.shutdown.return_value = None
        mock_instance.account_info.return_value = MagicMock(login=123, balance=10000, equity=10000, profit=0)
        mock_instance.symbol_info.return_value = MagicMock(point=0.00001, trade_tick_size=0.00001, trade_tick_value=1.0, volume_min=0.01, volume_max=100.0, volume_step=0.01)
        mock_instance.symbol_info_tick.return_value = MagicMock(last=1.2345)
        mock_instance.positions_get.return_value = []
        mock_instance.order_send.return_value = MagicMock(retcode=MetaTrader5.TRADE_RETCODE_DONE, order=12345)
        yield mock_instance

@pytest.fixture
def mt5_connector(mock_mt5linux):
    return MT5Connector("localhost", 18812, 12345, "password", "server")

def test_connect_success(mt5_connector):
    assert mt5_connector.connect() is True
    mt5_connector.mt5.initialize.assert_called_once_with(login=12345, password="password", server="server")

def test_connect_failure(mt5_connector):
    mt5_connector.mt5.initialize.return_value = False
    assert mt5_connector.connect() is False

def test_disconnect(mt5_connector):
    mt5_connector.disconnect()
    mt5_connector.mt5.shutdown.assert_called_once()

def test_place_order(mt5_connector):
    result = mt5_connector.place_order("EURUSD", "BUY", 0.01, 1.2345)
    assert result.retcode == MetaTrader5.TRADE_RETCODE_DONE
    mt5_connector.mt5.order_send.assert_called_once()

def test_get_positions(mt5_connector):
    positions = mt5_connector.get_positions()
    assert isinstance(positions, list)
    mt5_connector.mt5.positions_get.assert_called_once()

def test_get_historical_data(mt5_connector):
    mt5_connector.mt5.copy_rates_from_pos.return_value = [(1,2,3,4,5)] # Mock some data
    df = mt5_connector.get_historical_data("EURUSD", mt5_connector.mt5.TIMEFRAME_M1, 10)
    assert not df.empty
    mt5_connector.mt5.copy_rates_from_pos.assert_called_once()

def test_account_info(mt5_connector):
    info = mt5_connector.account_info()
    assert info.login == 123

def test_symbol_info(mt5_connector):
    info = mt5_connector.symbol_info("EURUSD")
    assert info.point == 0.00001

def test_symbol_info_tick(mt5_connector):
    info = mt5_connector.symbol_info_tick("EURUSD")
    assert info.last == 1.2345
