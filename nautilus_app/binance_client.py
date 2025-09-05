from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException

from .config import Settings


_TESTNET_API_URL = "https://testnet.binance.vision"


@dataclass
class Balance:
    asset: str
    free: float
    locked: float
    avg_price: float = 0.0
    current_price: float = 0.0
    profit_rate: float = 0.0
    profit_rate_no_fee: float = 0.0


class BinanceService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = Client(settings.api_key, settings.api_secret, testnet=settings.use_testnet)
        if settings.use_testnet:
            # Ensure correct testnet base URL for safety across client versions
            self.client.API_URL = _TESTNET_API_URL

    def get_price(self, symbol: str) -> float:
        data = self.client.get_symbol_ticker(symbol=symbol.upper())
        return float(data["price"])  # type: ignore[index]

    def get_balances(self) -> List[Balance]:
        account = self.client.get_account()
        balances: List[Balance] = []
        
        for b in account.get("balances", []):
            free = float(b.get("free", 0))
            locked = float(b.get("locked", 0))
            if free > 0 or locked > 0:
                asset = b["asset"]
                total_qty = free + locked
                
                # 평균단가 계산
                avg_price = self._calculate_avg_price(asset, total_qty)
                
                # 현재가 조회 (USDT 페어만)
                current_price = 0.0
                if asset != "USDT":
                    try:
                        symbol = f"{asset}USDT"
                        current_price = self.get_price(symbol)
                    except:
                        pass
                elif asset == "USDT":
                    current_price = 1.0
                
                # 수익률 계산
                profit_rate = 0.0
                profit_rate_no_fee = 0.0
                if avg_price > 0 and current_price > 0:
                    profit_rate = ((current_price - avg_price) / avg_price) * 100
                    # 수수료 제외 수익률 (0.1% 수수료 가정)
                    profit_rate_no_fee = profit_rate - 0.1
                
                balances.append(Balance(
                    asset=asset,
                    free=free,
                    locked=locked,
                    avg_price=avg_price,
                    current_price=current_price,
                    profit_rate=profit_rate,
                    profit_rate_no_fee=profit_rate_no_fee
                ))
        return balances
    
    def _calculate_avg_price(self, asset: str, quantity: float) -> float:
        if (asset == "USDT"):
            return 0.0
        
        """거래 내역을 통해 평균단가 계산"""
        try:
            # 최근 100개 거래 내역 조회
            trades = self.client.get_my_trades(symbol=f"{asset}USDT", limit=100)
            
            if not trades:
                return 0.0
            
            # 시간순으로 정렬 (오래된 것부터)
            trades.sort(key=lambda x: x["time"])
            
            total_cost = 0.0
            total_qty = 0.0
            
            for trade in trades:
                trade_qty = float(trade["qty"])
                trade_price = float(trade["price"])
                is_buyer = trade["isBuyer"]
                
                if is_buyer:  # 매수
                    total_cost += trade_qty * trade_price
                    total_qty += trade_qty
                else:  # 매도
                    # 매도 시에는 비례적으로 평균단가 조정
                    if total_qty > 0:
                        sold_ratio = trade_qty / total_qty
                        total_cost -= total_cost * sold_ratio
                        total_qty -= trade_qty
            
            # 현재 보유 수량과 계산된 수량이 비슷한지 확인
            if total_qty > 0 and abs(total_qty - quantity) / quantity < 0.1:  # 10% 오차 허용
                return total_cost / total_qty
            return 0.0
            
        except Exception as e:
            print(f"평균단가 계산 오류: asset={asset}, quantity={quantity}, error={e}")
            return 0.0

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        if symbol:
            return self.client.get_open_orders(symbol=symbol.upper())
        return self.client.get_open_orders()

    def market_buy(self, symbol: str, quantity: float) -> Dict:
        return self.client.order_market_buy(symbol=symbol.upper(), quantity=quantity)

    def market_sell(self, symbol: str, quantity: float) -> Dict:
        return self.client.order_market_sell(symbol=symbol.upper(), quantity=quantity)

    def cancel_order(self, symbol: str, order_id: int) -> Dict:
        return self.client.cancel_order(symbol=symbol.upper(), orderId=order_id)

    def limit_buy(self, symbol: str, quantity: float, price: float) -> Dict:
        """지정가 매수 주문"""
        return self.client.order_limit_buy(symbol=symbol.upper(), quantity=quantity, price=price)

    def limit_sell(self, symbol: str, quantity: float, price: float) -> Dict:
        """지정가 매도 주문"""
        return self.client.order_limit_sell(symbol=symbol.upper(), quantity=quantity, price=price)

    def stop_loss(self, symbol: str, quantity: float, stop_price: float) -> Dict:
        """손절매 주문 (Stop Loss)"""
        return self.client.order_stop_loss(symbol=symbol.upper(), quantity=quantity, stopPrice=stop_price)

    def take_profit(self, symbol: str, quantity: float, price: float) -> Dict:
        """익절매 주문 (Take Profit)"""
        return self.client.order_limit_sell(symbol=symbol.upper(), quantity=quantity, price=price)


