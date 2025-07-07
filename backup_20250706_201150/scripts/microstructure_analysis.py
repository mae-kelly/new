#!/usr/bin/env python3

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
    """Order book snapshot data"""
    timestamp: float
    bids: List[Tuple[float, float]]  # (price, size)
    asks: List[Tuple[float, float]]  # (price, size)
    mid_price: float
    spread: float
    depth: float

@dataclass
class MarketImpactModel:
    """Market impact estimation"""
    permanent_impact: float
    temporary_impact: float
    liquidity_measure: float
    optimal_execution_time: float

class AdvancedMicrostructureAnalyzer:
    """Advanced market microstructure analysis"""
    
    def __init__(self):
        self.order_book_history = []
        self.trade_history = []
    
    async def get_level2_data(self, token_address: str) -> OrderBookSnapshot:
        """Fetch Level II order book data"""
        # This would integrate with DEX APIs that provide order book data
        # For Solana: Serum, Mango, or other DEX APIs
        
        # Placeholder implementation
        timestamp = asyncio.get_event_loop().time()
        
        # Mock order book data
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
        """Estimate market impact using Almgren-Chriss model"""
        
        # Calculate liquidity measures
        bid_liquidity = sum(size for _, size in order_book.bids[:10])
        ask_liquidity = sum(size for _, size in order_book.asks[:10])
        total_liquidity = bid_liquidity + ask_liquidity
        
        # Estimate temporary impact (proportional to order size / liquidity)
        temporary_impact = order_size / total_liquidity * order_book.spread
        
        # Estimate permanent impact (smaller, square root of size)
        permanent_impact = 0.1 * np.sqrt(order_size / total_liquidity) * order_book.spread
        
        # Optimal execution time (balance impact vs. risk)
        volatility = 0.02  # Daily volatility estimate
        optimal_time = np.sqrt(temporary_impact / (volatility**2)) * 86400  # seconds
        
        return MarketImpactModel(
            permanent_impact=permanent_impact,
            temporary_impact=temporary_impact,
            liquidity_measure=total_liquidity,
            optimal_execution_time=min(optimal_time, 3600)  # Max 1 hour
        )
    
    def calculate_probability_of_informed_trading(self, trades: List[Dict]) -> float:
        """Calculate PIN (Probability of Informed Trading)"""
        if len(trades) < 100:
            return 0.5  # Default
        
        # Classify trades as buy/sell based on price movement
        buys = sum(1 for trade in trades if trade.get('side') == 'buy')
        sells = len(trades) - buys
        
        # Simplified PIN calculation
        if buys + sells == 0:
            return 0.5
        
        imbalance = abs(buys - sells) / (buys + sells)
        
        # Higher imbalance suggests more informed trading
        pin = min(imbalance * 2, 0.9)  # Cap at 90%
        
        return pin
    
    def detect_front_running_opportunities(self, pending_txs: List[Dict]) -> List[Dict]:
        """Detect potential front-running opportunities in mempool"""
        opportunities = []
        
        # Look for large transactions that could be front-run
        for tx in pending_txs:
            if tx.get('amount', 0) > 10000:  # Large trade threshold
                # Calculate potential profit from front-running
                estimated_impact = tx['amount'] * 0.001  # 0.1% impact estimate
                
                if estimated_impact > 0.01:  # 1% minimum profit
                    opportunities.append({
                        'tx_hash': tx.get('hash'),
                        'estimated_profit': estimated_impact,
                        'gas_cost': tx.get('gas_price', 0) * 1.1,  # Need to pay more gas
                        'risk_score': 0.3  # Front-running risk
                    })
        
        return opportunities
    
    def calculate_effective_spread(self, trades: List[Dict]) -> float:
        """Calculate effective spread from trade data"""
        if len(trades) < 10:
            return 0.01  # Default 1%
        
        # Calculate price improvements relative to quoted spread
        improvements = []
        
        for i in range(1, len(trades)):
            price_change = abs(trades[i]['price'] - trades[i-1]['price'])
            time_diff = trades[i]['timestamp'] - trades[i-1]['timestamp']
            
            if time_diff > 0:
                improvements.append(price_change / trades[i-1]['price'])
        
        return np.mean(improvements) if improvements else 0.01
    
    def identify_arbitrage_opportunities(self, prices_across_exchanges: Dict[str, float]) -> List[Dict]:
        """Identify cross-exchange arbitrage opportunities"""
        opportunities = []
        
        exchanges = list(prices_across_exchanges.keys())
        prices = list(prices_across_exchanges.values())
        
        if len(exchanges) < 2:
            return opportunities
        
        # Find price differences
        for i in range(len(exchanges)):
            for j in range(i + 1, len(exchanges)):
                price_diff = abs(prices[i] - prices[j])
                avg_price = (prices[i] + prices[j]) / 2
                
                if avg_price > 0:
                    spread_pct = price_diff / avg_price
                    
                    # Profitable if spread > transaction costs (estimate 0.5%)
                    if spread_pct > 0.005:
                        buy_exchange = exchanges[i] if prices[i] < prices[j] else exchanges[j]
                        sell_exchange = exchanges[j] if prices[i] < prices[j] else exchanges[i]
                        
                        opportunities.append({
                            'buy_exchange': buy_exchange,
                            'sell_exchange': sell_exchange,
                            'profit_pct': spread_pct - 0.005,  # Subtract estimated costs
                            'buy_price': min(prices[i], prices[j]),
                            'sell_price': max(prices[i], prices[j])
                        })
        
        return sorted(opportunities, key=lambda x: x['profit_pct'], reverse=True)

def main():
    """Test microstructure analysis components"""
    print("🔬 Testing Microstructure Analysis...")
    
    analyzer = AdvancedMicrostructureAnalyzer()
    
    # Test market impact estimation
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
    
    # Test arbitrage detection
    mock_prices = {
        'raydium': 1.000,
        'orca': 1.015,
        'serum': 0.995
    }
    
    arbitrage_ops = analyzer.identify_arbitrage_opportunities(mock_prices)
    print(f"✅ Found {len(arbitrage_ops)} arbitrage opportunities")
    for op in arbitrage_ops:
        print(f"   Buy on {op['buy_exchange']}, sell on {op['sell_exchange']}: {op['profit_pct']:.2%} profit")

if __name__ == "__main__":
    main()
