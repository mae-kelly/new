import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import asyncio
import aiohttp
from dataclasses import dataclass
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
@dataclass
class OrderBookSnapshot:
    permanent_impact: float
    temporary_impact: float
    liquidity_measure: float
    optimal_execution_time: float
class AdvancedMicrostructureAnalyzer:
        timestamp = asyncio.get_event_loop().time()
        bids = [(0.995, 1000), (0.990, 2000), (0.985, 1500)]
        asks = [(1.005, 1200), (1.010, 1800), (1.015, 1000)]
        mid_price = (bids[0][0] + asks[0][0]) / 2
        spread = asks[0][0] - bids[0][0]
        depth = sum(size for _, size in bids[:5]) + sum(size for _, size in asks[:5])
        return OrderBookSnapshot(
            timestamp=timestamp,
            bids=bids,
            asks=asks,
            mid_price=mid_price,
            spread=spread,
            depth=depth
        )
    def estimate_market_impact(self, order_size: float, 
                             order_book: OrderBookSnapshot) -> MarketImpactModel:
        if len(trades) < 100:
            return 0.5
        buys = sum(1 for trade in trades if trade.get('side') == 'buy')
        sells = len(trades) - buys
        if buys + sells == 0:
            return 0.5
        imbalance = abs(buys - sells) / (buys + sells)
        pin = min(imbalance * 2, 0.9)
        return pin
    def detect_front_running_opportunities(self, pending_txs: List[Dict]) -> List[Dict]:
        if len(trades) < 10:
            return 0.01
        improvements = []
        for i in range(1, len(trades)):
            price_change = abs(trades[i]['price'] - trades[i-1]['price'])
            time_diff = trades[i]['timestamp'] - trades[i-1]['timestamp']
            if time_diff > 0:
                improvements.append(price_change / trades[i-1]['price'])
        return np.mean(improvements) if improvements else 0.01
    def identify_arbitrage_opportunities(self, prices_across_exchanges: Dict[str, float]) -> List[Dict]:
    print("🔬 Testing Microstructure Analysis...")
    analyzer = AdvancedMicrostructureAnalyzer()
    mock_order_book = OrderBookSnapshot(
        timestamp=1234567890,
        bids=[(0.995, 1000), (0.990, 2000), (0.985, 1500)],
        asks=[(1.005, 1200), (1.010, 1800), (1.015, 1000)],
        mid_price=1.0,
        spread=0.01,
        depth=8500
    )
    impact = analyzer.estimate_market_impact(500, mock_order_book)
    print(f"✅ Market impact for 500 tokens: {impact.temporary_impact:.4f}")
    print(f"✅ Optimal execution time: {impact.optimal_execution_time:.0f} seconds")
    mock_prices = {
        'raydium': 1.000,
        'orca': 1.015,
        'serum': 0.995
    }
    arbitrage_ops = analyzer.identify_arbitrage_opportunities(mock_prices)
    print(f"✅ Found {len(arbitrage_ops)} arbitrage opportunities")
    for op in arbitrage_ops:
        print(f"   Buy on {op['buy_exchange']}, sell on {op['sell_exchange']}: {op['profit_pct']:.2
if __name__ == "__main__":
    main()
