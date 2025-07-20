from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel

class OrderType(Enum):
    """Order types"""
    BUY = "BUY"
    SELL = "SELL"
    BUY_LIMIT = "BUY_LIMIT"
    SELL_LIMIT = "SELL_LIMIT"
    BUY_STOP = "BUY_STOP"
    SELL_STOP = "SELL_STOP"

class SignalType(Enum):
    """Trading signal types"""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    WEAK_BUY = "WEAK_BUY"
    HOLD = "HOLD"
    WEAK_SELL = "WEAK_SELL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"

class PositionSide(Enum):
    """Position sides"""
    LONG = "LONG"
    SHORT = "SHORT"

@dataclass
class OrderRequest:
    """Trading order request"""
    symbol: str
    order_type: OrderType
    volume: float
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    deviation: int = 10
    magic_number: int = 123456
    comment: str = ""
    position_id: Optional[int] = None

@dataclass
class OrderResult:
    """Trading order result"""
    success: bool
    order_id: Optional[int] = None
    ticket: Optional[int] = None
    price: Optional[float] = None
    volume: Optional[float] = None
    error: Optional[str] = None
    comment: Optional[str] = None

@dataclass
class Position:
    """Trading position"""
    ticket: int
    symbol: str
    type: int  # 0=BUY, 1=SELL
    volume: float
    open_price: float
    current_price: float
    swap: float
    profit: float
    open_time: datetime
    magic_number: int
    comment: str
    
    @property
    def side(self) -> PositionSide:
        return PositionSide.LONG if self.type == 0 else PositionSide.SHORT
    
    @property
    def unrealized_pnl(self) -> float:
        return self.profit

@dataclass
class AccountInfo:
    """Account information"""
    login: int
    balance: float
    equity: float
    margin: float
    free_margin: float
    margin_level: float
    currency: str

@dataclass
class MarketInfo:
    """Market/Symbol information"""
    symbol: str
    bid: float
    ask: float
    spread: int
    point: float
    digits: int
    tick_value: float
    tick_size: float
    contract_size: float
    min_lot: float
    max_lot: float
    lot_step: float
    
    @property
    def spread_in_price(self) -> float:
        return self.spread * self.point

class TradeSignal(BaseModel):
    """Trading signal from ML models"""
    timestamp: datetime
    symbol: str
    signal_type: SignalType
    confidence: float
    predicted_price: float
    current_price: float
    price_change_percent: float
    horizon: str
    
    # Signal details
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    
    # Model info
    model_name: str
    model_confidence: float
    
    # Risk metrics
    volatility: float
    max_position_size: float

class TradeSetup(BaseModel):
    """Complete trade setup"""
    signal: TradeSignal
    position_size: float
    order_type: OrderType
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_amount: float
    expected_return: float
    risk_reward_ratio: float
    
    # Validation flags
    risk_approved: bool = False
    size_approved: bool = False
    timing_approved: bool = False
