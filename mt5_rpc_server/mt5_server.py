import rpyc
from rpyc.utils.server import ThreadedServer
import MetaTrader5 as mt5
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MT5Service(rpyc.Service):
    def exposed_initialize(self, *args, **kwargs):
        logger.info(f"MT5Service: initialize({args}, {kwargs})")
        return mt5.initialize(*args, **kwargs)
    
    def exposed_login(self, *args, **kwargs):
        logger.info(f"MT5Service: login({args}, {kwargs})")
        return mt5.login(*args, **kwargs)
    
    def exposed_shutdown(self):
        logger.info("MT5Service: shutdown()")
        return mt5.shutdown()
    
    def exposed_terminal_info(self):
        logger.info("MT5Service: terminal_info()")
        return mt5.terminal_info()
    
    def exposed_account_info(self):
        logger.info("MT5Service: account_info()")
        return mt5.account_info()
    
    def exposed_copy_rates_from_pos(self, *args, **kwargs):
        logger.info(f"MT5Service: copy_rates_from_pos({args}, {kwargs})")
        return mt5.copy_rates_from_pos(*args, **kwargs)
    
    def exposed_copy_ticks_from(self, *args, **kwargs):
        logger.info(f"MT5Service: copy_ticks_from({args}, {kwargs})")
        return mt5.copy_ticks_from(*args, **kwargs)
    
    def exposed_order_send(self, *args, **kwargs):
        logger.info(f"MT5Service: order_send({args}, {kwargs})")
        return mt5.order_send(*args, **kwargs)

    def exposed_positions_get(self, *args, **kwargs):
        logger.info(f"MT5Service: positions_get({args}, {kwargs})")
        return mt5.positions_get(*args, **kwargs)

    def exposed_symbol_info(self, *args, **kwargs):
        logger.info(f"MT5Service: symbol_info({args}, {kwargs})")
        return mt5.symbol_info(*args, **kwargs)

    def exposed_symbol_info_tick(self, *args, **kwargs):
        logger.info(f"MT5Service: symbol_info_tick({args}, {kwargs})")
        return mt5.symbol_info_tick(*args, **kwargs)

if __name__ == "__main__":
    # Default port for mt5linux RPC server is 18812
    port = 18812
    server = ThreadedServer(MT5Service, port=port)
    logger.info(f"MT5 RPC Server starting on port {port}...")
    server.start()
