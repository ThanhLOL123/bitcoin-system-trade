import asyncio
import logging
import signal
import sys
from typing import Dict
from datetime import datetime
import json

from .strategy.strategy_engine import StrategyEngine
from .monitoring.trading_metrics import TradingMetrics
from .mt5_connector import MT5Connector
from .portfolio.portfolio_manager import PortfolioManager
from .risk_management.risk_manager import RiskManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class TradingBot:
    """Main trading bot application"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.mt5_connector = MT5Connector(
            config['mt5']['mt5_rpc_host'],
            config['mt5']['mt5_rpc_port'],
            config['mt5']['login'],
            config['mt5']['password'],
            config['mt5']['server']
        )
        self.portfolio_manager = PortfolioManager(config['portfolio'])
        self.strategy_engine = StrategyEngine(config, self.mt5_connector, self.portfolio_manager)
        self.metrics = TradingMetrics()
        self.is_running = False
        
        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.is_running = False
    
    async def start(self):
        """Start the trading bot"""
        try:
            logger.info("Starting Bitcoin Trading Bot")
            logger.info(f"Configuration: {json.dumps(self.config, indent=2, default=str)}")
            
            # Validate configuration
            self._validate_config()

            # Connect to MT5
            if not self.mt5_connector.connect():
                raise Exception("Failed to connect to MT5")
            
            # Start strategy engine
            self.is_running = True
            await self.strategy_engine.start()
            
        except Exception as e:
            logger.error(f"Failed to start trading bot: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the trading bot"""
        try:
            logger.info("Stopping trading bot...")
            
            self.is_running = False
            
            # Stop strategy engine
            await self.strategy_engine.stop()

            # Disconnect from MT5
            self.mt5_connector.disconnect()
            
            # Generate final performance report
            await self._generate_final_report()
            
            logger.info("Trading bot stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping trading bot: {e}")
    
    def _validate_config(self):
        """Validate trading configuration"""
        required_keys = ['mt5', 'risk', 'portfolio', 'strategies_enabled']
        
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
        
        # Validate MT5 config
        mt5_config = self.config['mt5']
        required_mt5_keys = ['login', 'password', 'server', 'mt5_rpc_host', 'mt5_rpc_port']
        
        for key in required_mt5_keys:
            if key not in mt5_config:
                raise ValueError(f"Missing required MT5 config key: {key}")
        
        # Validate risk parameters
        risk_config = self.config['risk']
        if risk_config.get('max_daily_loss_percent', 0) <= 0:
            raise ValueError("Invalid max_daily_loss_percent")
        
        if risk_config.get('max_position_size_percent', 0) <= 0:
            raise ValueError("Invalid max_position_size_percent")
        
        logger.info("Configuration validated successfully")
    
    async def _generate_final_report(self):
        """Generate final performance report"""
        try:
            performance = self.portfolio_manager.get_performance_summary()
            
            report = {
                'session_end_time': datetime.now().isoformat(),
                'performance_summary': performance,
                'final_metrics': self.metrics.get_metrics() # Assuming get_metrics() exists in TradingMetrics
            }
            
            # Save report to file
            filename = f"trading_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Final performance report saved to {filename}")
            logger.info(f"Session summary: {json.dumps(performance, indent=2)}")
            
        except Exception as e:
            logger.error(f"Error generating final report: {e}")

async def main():
    from .config import TRADING_CONFIG
    bot = TradingBot(TRADING_CONFIG)
    try:
        await bot.start()
    except asyncio.CancelledError:
        logger.info("Bot task cancelled.")
    finally:
        await bot.stop()

if __name__ == "__main__":
    asyncio.run(main())
