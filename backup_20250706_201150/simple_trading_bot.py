#!/usr/bin/env python3
"""
SIMPLIFIED WORKING CRYPTO TRADING BOT
No complex dependencies - just working code
"""

import asyncio
import json
import time
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os

# Simple HTTP client instead of complex libraries
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    import requests
    AIOHTTP_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SimpleTradingBot:
    """
    Simplified trading bot that actually works
    """
    
    def __init__(self, initial_balance: float = 10.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.positions = {}
        self.trade_history = []
        self.running = True
        
        # Initialize database
        self.init_database()
        
        logger.info(f"🚀 Simple Trading Bot Initialized")
        logger.info(f"💰 Initial Balance: ${self.initial_balance}")
        logger.info(f"📋 Mode: PAPER TRADING (No real money)")
    
    def init_database(self):
        """Initialize SQLite database"""
        self.conn = sqlite3.connect('simple_bot.db')
        cursor = self.conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                token_address TEXT,
                action TEXT,
                amount REAL,
                price REAL,
                profit_loss REAL,
                balance_after REAL
            )
        ''')
        
        self.conn.commit()
        logger.info("📊 Database initialized")
    
    def simulate_token_discovery(self) -> Dict:
        """Simulate finding a new token"""
        import random
        
        # Generate fake token data
        token_address = f"{''.join(random.choices('ABCDEFGHJKMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz123456789', k=44))}"
        
        return {
            'address': token_address,
            'name': f"MemeCoin{random.randint(1000, 9999)}",
            'symbol': f"MEME{random.randint(100, 999)}",
            'price': random.uniform(0.000001, 0.0001),
            'is_safe': random.choice([True, False, False, True, True]),  # 60% safe
            'confidence': random.uniform(0.3, 0.9)
        }
    
    def analyze_token(self, token_data: Dict) -> bool:
        """Simple token analysis"""
        # Simple criteria for buying
        if (token_data['is_safe'] and 
            token_data['confidence'] > 0.6 and 
            token_data['price'] < 0.00005):
            return True
        return False
    
    def execute_buy(self, token_data: Dict) -> bool:
        """Execute simulated buy"""
        position_size = min(self.current_balance * 0.25, 2.5)  # 25% or max $2.50
        
        if position_size < 0.5:  # Minimum $0.50
            return False
        
        # Execute trade
        self.current_balance -= position_size
        
        # Store position
        self.positions[token_data['address']] = {
            'entry_price': token_data['price'],
            'amount': position_size / token_data['price'],
            'invested': position_size,
            'entry_time': datetime.now(),
            'name': token_data['name']
        }
        
        # Record trade
        self.record_trade(token_data['address'], 'BUY', position_size, 
                         token_data['price'], 0, self.current_balance)
        
        logger.info(f"📈 BOUGHT: {token_data['name']} (${position_size:.2f})")
        return True
    
    def check_exit_conditions(self, token_address: str, position: Dict) -> tuple:
        """Check if we should sell a position"""
        import random
        
        # Simulate price movement
        entry_price = position['entry_price']
        current_price = entry_price * random.uniform(0.1, 100.0)  # Wild price swings
        
        multiplier = current_price / entry_price
        time_held = datetime.now() - position['entry_time']
        
        # Exit conditions
        if multiplier >= 50:
            return True, "PROFIT_50x", multiplier, current_price
        elif multiplier >= 25:
            return True, "PROFIT_25x", multiplier, current_price
        elif multiplier >= 10:
            return True, "PROFIT_10x", multiplier, current_price
        elif multiplier >= 5:
            return True, "PROFIT_5x", multiplier, current_price
        elif multiplier <= 0.5:
            return True, "STOP_LOSS", multiplier, current_price
        elif time_held > timedelta(hours=2):  # 2 hour max hold
            return True, "TIME_LIMIT", multiplier, current_price
        
        return False, "HOLD", multiplier, current_price
    
    def execute_sell(self, token_address: str, reason: str, multiplier: float, price: float):
        """Execute simulated sell"""
        if token_address not in self.positions:
            return
        
        position = self.positions[token_address]
        proceeds = position['invested'] * multiplier
        profit = proceeds - position['invested']
        
        self.current_balance += proceeds
        
        # Record trade
        self.record_trade(token_address, 'SELL', proceeds, price, profit, self.current_balance)
        
        logger.info(f"💰 SOLD: {position['name']} | {multiplier:.2f}x | Profit: ${profit:+.2f}")
        logger.info(f"💵 New Balance: ${self.current_balance:.2f}")
        
        # Remove position
        del self.positions[token_address]
    
    def record_trade(self, token_address: str, action: str, amount: float, 
                    price: float, profit: float, balance_after: float):
        """Record trade in database"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO trades (timestamp, token_address, action, amount, price, profit_loss, balance_after)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (datetime.now().isoformat(), token_address, action, amount, price, profit, balance_after))
        self.conn.commit()
    
    def print_status(self):
        """Print current status"""
        total_return = ((self.current_balance / self.initial_balance) - 1) * 100
        progress_to_10k = (self.current_balance / 10000) * 100
        
        logger.info("=" * 60)
        logger.info(f"💰 Balance: ${self.current_balance:.2f} | Return: {total_return:+.1f}%")
        logger.info(f"🎯 Progress to $10K: {progress_to_10k:.3f}%")
        logger.info(f"📊 Active Positions: {len(self.positions)}")
        
        if self.current_balance >= 10000:
            logger.info("🎉 TARGET ACHIEVED! $10K+ REACHED!")
        elif self.current_balance >= 1000:
            logger.info("🚀 Getting close! Over $1K!")
        elif self.current_balance >= 100:
            logger.info("📈 Good progress! Over $100!")
        
        logger.info("=" * 60)
    
    async def run_simulation(self, duration_minutes: int = 60):
        """Run the trading simulation"""
        logger.info(f"🚀 Starting {duration_minutes}-minute trading simulation...")
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        iteration = 0
        
        while datetime.now() < end_time and self.running:
            try:
                iteration += 1
                
                # Check existing positions
                for token_address, position in list(self.positions.items()):
                    should_sell, reason, multiplier, price = self.check_exit_conditions(token_address, position)
                    
                    if should_sell:
                        self.execute_sell(token_address, reason, multiplier, price)
                
                # Look for new opportunities
                if len(self.positions) < 3 and self.current_balance > 1.0:  # Max 3 positions
                    token_data = self.simulate_token_discovery()
                    
                    if self.analyze_token(token_data):
                        success = self.execute_buy(token_data)
                        if success:
                            logger.info(f"✅ New position opened")
                
                # Status update every 10 iterations
                if iteration % 10 == 0:
                    self.print_status()
                
                # Check if we've reached target
                if self.current_balance >= 10000:
                    logger.info("🎉 TARGET REACHED! Stopping bot...")
                    break
                
                # Small delay
                await asyncio.sleep(1)
                
            except KeyboardInterrupt:
                logger.info("🛑 Bot stopped by user")
                self.running = False
                break
            except Exception as e:
                logger.error(f"❌ Error in main loop: {e}")
                await asyncio.sleep(5)
        
        # Final status
        self.print_status()
        
        # Show results
        total_return = ((self.current_balance / self.initial_balance) - 1) * 100
        logger.info(f"📊 FINAL RESULTS:")
        logger.info(f"💰 Final Balance: ${self.current_balance:.2f}")
        logger.info(f"📈 Total Return: {total_return:+.1f}%")
        
        if self.current_balance >= 10000:
            logger.info("🏆 SUCCESS! Target achieved!")
        else:
            logger.info(f"🎯 Progress: {(self.current_balance/10000)*100:.2f}% to $10K target")

# Main execution
async def main():
    print("🚀 SIMPLE CRYPTO TRADING BOT")
    print("💡 Simulated trading - No real money involved")
    print("🎯 Target: Turn $10 into $10,000")
    print()
    
    # Get user input
    try:
        balance = float(input("💰 Starting balance (default $10): ") or "10")
        duration = int(input("⏱️  Simulation duration in minutes (default 30): ") or "30")
    except ValueError:
        balance = 10.0
        duration = 30
    
    print(f"🎮 Starting simulation with ${balance} for {duration} minutes...")
    print("Press Ctrl+C to stop early")
    print()
    
    # Create and run bot
    bot = SimpleTradingBot(initial_balance=balance)
    await bot.run_simulation(duration_minutes=duration)

if __name__ == "__main__":
    asyncio.run(main())
