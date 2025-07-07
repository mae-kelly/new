#!/usr/bin/env python3

import os
import sys
import json
import time
import asyncio
import logging
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

# Crypto libraries
import base58
from solana.rpc.async_api import AsyncClient
from solana.keypair import Keypair
from solders.pubkey import Pubkey
from solders.transaction import VersionedTransaction
from solders.instruction import Instruction
from solders.system_program import TransferParams, transfer
import aiohttp

# Import our modules
sys.path.append('scripts')
from contract_analyzer import SolanaContractAnalyzer, ContractAnalysisResult
from token_discovery import TokenDiscovery, TokenDatabase
from ml_system import AdvancedMLSystem, TrainingEngine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading_engine.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TradingConfig:
    """Trading configuration parameters"""
    max_position_size: float = 0.1      # Max 10% of portfolio per trade
    min_confidence_score: float = 0.7   # ML confidence threshold
    max_slippage: float = 0.05          # 5% max slippage
    stop_loss_pct: float = 0.5          # 50% stop loss
    take_profit_pct: float = 5.0        # 500% take profit
    max_hold_time: int = 3600           # 1 hour max hold
    min_liquidity: float = 2000         # $2k min liquidity
    max_sell_tax: float = 0.1           # 10% max sell tax
    gas_limit: int = 200000             # Gas limit for transactions
    priority_fee: int = 5000            # Priority fee in microlamports
    max_daily_trades: int = 50          # Daily trade limit
    risk_multiplier: float = 1.0        # Risk adjustment factor

@dataclass
class TradePosition:
    """Active trading position"""
    token_address: str
    entry_price: float
    quantity: float
    entry_time: float
    stop_loss: float
    take_profit: float
    confidence_score: float
    transaction_hash: str = ""
    current_price: float = 0
    unrealized_pnl: float = 0
    status: str = "OPEN"  # OPEN, CLOSED, STOPPED

@dataclass
class TradeResult:
    """Completed trade result"""
    token_address: str
    entry_price: float
    exit_price: float
    quantity: float
    entry_time: float
    exit_time: float
    pnl: float
    pnl_pct: float
    reason: str  # TAKE_PROFIT, STOP_LOSS, TIME_LIMIT, MANUAL
    confidence_score: float
    fees_paid: float = 0

class PortfolioManager:
    """Portfolio and risk management"""
    
    def __init__(self, initial_balance: float = 10.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.positions: Dict[str, TradePosition] = {}
        self.trade_history: List[TradeResult] = []
        self.daily_trades = 0
        self.last_trade_date = datetime.now().date()
        
    def get_available_balance(self) -> float:
        """Get available balance for trading"""
        # Calculate total position value
        position_value = sum(pos.quantity * pos.current_price for pos in self.positions.values())
        return max(0, self.current_balance - position_value)
    
    def calculate_position_size(self, price: float, confidence: float, config: TradingConfig) -> float:
        """Calculate optimal position size using Kelly Criterion"""
        available = self.get_available_balance()
        max_position_value = available * config.max_position_size
        
        # Kelly Criterion adjustment
        win_rate = self._estimate_win_rate()
        avg_win = self._estimate_avg_win()
        avg_loss = self._estimate_avg_loss()
        
        if avg_loss == 0:
            kelly_fraction = 0.1  # Conservative default
        else:
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
        
        # Adjust by confidence and risk multiplier
        adjusted_fraction = kelly_fraction * confidence * config.risk_multiplier
        position_value = min(max_position_value, available * adjusted_fraction)
        
        return position_value / price if price > 0 else 0
    
    def _estimate_win_rate(self) -> float:
        """Estimate win rate from trade history"""
        if len(self.trade_history) < 10:
            return 0.6  # Conservative estimate
        
        wins = sum(1 for trade in self.trade_history[-50:] if trade.pnl > 0)
        return wins / min(len(self.trade_history), 50)
    
    def _estimate_avg_win(self) -> float:
        """Estimate average win from profitable trades"""
        wins = [trade.pnl_pct for trade in self.trade_history[-50:] if trade.pnl > 0]
        return np.mean(wins) if wins else 0.5
    
    def _estimate_avg_loss(self) -> float:
        """Estimate average loss from losing trades"""
        losses = [-trade.pnl_pct for trade in self.trade_history[-50:] if trade.pnl < 0]
        return np.mean(losses) if losses else 0.3
    
    def add_position(self, position: TradePosition) -> bool:
        """Add new position to portfolio"""
        if position.token_address in self.positions:
            logger.warning(f"Position already exists for {position.token_address}")
            return False
        
        self.positions[position.token_address] = position
        return True
    
    def close_position(self, token_address: str, exit_price: float, reason: str) -> Optional[TradeResult]:
        """Close position and record trade result"""
        if token_address not in self.positions:
            return None
        
        position = self.positions[token_address]
        
        # Calculate PnL
        pnl = (exit_price - position.entry_price) * position.quantity
        pnl_pct = (exit_price / position.entry_price - 1) if position.entry_price > 0 else 0
        
        # Create trade result
        trade_result = TradeResult(
            token_address=token_address,
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            entry_time=position.entry_time,
            exit_time=time.time(),
            pnl=pnl,
            pnl_pct=pnl_pct,
            reason=reason,
            confidence_score=position.confidence_score
        )
        
        # Update balance and remove position
        self.current_balance += pnl
        self.trade_history.append(trade_result)
        del self.positions[token_address]
        
        logger.info(f"Position closed: {token_address[:8]} - PnL: {pnl_pct:.2%} - Reason: {reason}")
        return trade_result
    
    def update_position_prices(self, price_updates: Dict[str, float]):
        """Update current prices for all positions"""
        for token_address, current_price in price_updates.items():
            if token_address in self.positions:
                position = self.positions[token_address]
                position.current_price = current_price
                position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
    
    def get_portfolio_summary(self) -> Dict:
        """Get portfolio performance summary"""
        total_pnl = sum(trade.pnl for trade in self.trade_history)
        total_return = (self.current_balance / self.initial_balance - 1) if self.initial_balance > 0 else 0
        
        winning_trades = [trade for trade in self.trade_history if trade.pnl > 0]
        losing_trades = [trade for trade in self.trade_history if trade.pnl < 0]
        
        return {
            'initial_balance': self.initial_balance,
            'current_balance': self.current_balance,
            'total_return': total_return,
            'total_pnl': total_pnl,
            'total_trades': len(self.trade_history),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(self.trade_history) if self.trade_history else 0,
            'avg_win': np.mean([trade.pnl_pct for trade in winning_trades]) if winning_trades else 0,
            'avg_loss': np.mean([trade.pnl_pct for trade in losing_trades]) if losing_trades else 0,
            'active_positions': len(self.positions),
            'available_balance': self.get_available_balance()
        }

class TradingEngine:
    """Advanced automated trading engine"""
    
    def __init__(self, private_key: str, config: TradingConfig = None, paper_trading: bool = False):
        self.config = config or TradingConfig()
        self.paper_trading = paper_trading
        self.portfolio = PortfolioManager()
        self.session = None
        
        # Initialize components
        self.db = TokenDatabase()
        self.contract_analyzer = None
        self.token_discovery = None
        self.ml_system = None
        
        # Solana setup
        if not paper_trading:
            try:
                self.keypair = Keypair.from_secret_key(private_key)
                self.client = AsyncClient("https://api.mainnet-beta.solana.com")
                logger.info(f"Trading wallet: {self.keypair.pubkey()}")
            except Exception as e:
                logger.error(f"Wallet setup failed: {str(e)}")
                raise
        else:
            self.keypair = None
            self.client = None
            logger.info("Paper trading mode enabled")
        
        # Trading state
        self.is_running = False
        self.last_scan_time = 0
        self.scan_interval = 300  # 5 minutes
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        self.contract_analyzer = await SolanaContractAnalyzer().__aenter__()
        self.token_discovery = await TokenDiscovery().__aenter__()
        
        # Load ML models if available
        if os.path.exists('models/ml_system.pkl'):
            with open('models/ml_system.pkl', 'rb') as f:
                self.ml_system = pickle.load(f)
            logger.info("ML models loaded successfully")
        else:
            logger.warning("No ML models found - trading without ML predictions")
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
        if self.contract_analyzer:
            await self.contract_analyzer.__aexit__(exc_type, exc_val, exc_tb)
        if self.token_discovery:
            await self.token_discovery.__aexit__(exc_type, exc_val, exc_tb)
        if self.client:
            await self.client.close()
    
    async def start_trading(self):
        """Start the automated trading loop"""
        logger.info("🚀 Starting automated trading engine...")
        self.is_running = True
        
        try:
            while self.is_running:
                cycle_start = time.time()
                
                # Check daily trade limit
                current_date = datetime.now().date()
                if current_date != self.portfolio.last_trade_date:
                    self.portfolio.daily_trades = 0
                    self.portfolio.last_trade_date = current_date
                
                if self.portfolio.daily_trades >= self.config.max_daily_trades:
                    logger.info(f"Daily trade limit reached ({self.config.max_daily_trades})")
                    await asyncio.sleep(3600)  # Wait 1 hour
                    continue
                
                # Main trading cycle
                await self._trading_cycle()
                
                # Calculate cycle time and sleep
                cycle_time = time.time() - cycle_start
                sleep_time = max(self.scan_interval - cycle_time, 30)  # Min 30s
                
                logger.info(f"Trading cycle completed in {cycle_time:.1f}s, sleeping {sleep_time:.1f}s")
                await asyncio.sleep(sleep_time)
        
        except KeyboardInterrupt:
            logger.info("🛑 Trading stopped by user")
        except Exception as e:
            logger.error(f"❌ Trading error: {str(e)}")
        finally:
            self.is_running = False
    
    async def _trading_cycle(self):
        """Execute one trading cycle"""
        try:
            # 1. Scan for new opportunities
            await self._scan_for_opportunities()
            
            # 2. Monitor existing positions
            await self._monitor_positions()
            
            # 3. Execute pending trades
            await self._execute_trades()
            
            # 4. Update portfolio metrics
            self._update_portfolio_metrics()
            
        except Exception as e:
            logger.error(f"Trading cycle error: {str(e)}")
    
    async def _scan_for_opportunities(self):
        """Scan for new trading opportunities"""
        if time.time() - self.last_scan_time < self.scan_interval:
            return
        
        logger.info("🔍 Scanning for trading opportunities...")
        
        # Get new tokens
        new_tokens = await self.token_discovery.scan_new_tokens(max_age_minutes=30)
        
        if not new_tokens:
            logger.info("No new tokens found")
            return
        
        # Analyze tokens in parallel
        analysis_tasks = []
        for token in new_tokens[:20]:  # Limit to 20 tokens per cycle
            task = self._analyze_trading_opportunity(token.address)
            analysis_tasks.append(task)
        
        results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
        
        # Process results
        opportunities = []
        for result in results:
            if isinstance(result, dict) and result.get('is_opportunity'):
                opportunities.append(result)
        
        logger.info(f"Found {len(opportunities)} trading opportunities")
        
        # Execute top opportunities
        sorted_opportunities = sorted(
            opportunities, 
            key=lambda x: x.get('combined_score', 0), 
            reverse=True
        )
        
        for opportunity in sorted_opportunities[:3]:  # Top 3 opportunities
            await self._consider_trade(opportunity)
        
        self.last_scan_time = time.time()
    
    async def _analyze_trading_opportunity(self, token_address: str) -> Dict:
        """Analyze a single token for trading opportunity"""
        try:
            # 1. Contract safety analysis
            contract_result = await self.contract_analyzer.analyze_token(token_address)
            
            if not contract_result.is_safe:
                return {
                    'token_address': token_address,
                    'is_opportunity': False,
                    'reason': 'Contract unsafe',
                    'contract_score': 0
                }
            
            # 2. Get whale activity
            whale_signals = await self.token_discovery.track_whale_activity([token_address])
            whale_activity = whale_signals.get(token_address, {})
            
            # 3. Get detailed metrics
            metrics = await self.token_discovery.get_token_detailed_metrics(token_address)
            
            # 4. ML prediction (if available)
            ml_score = 0.5  # Default
            if self.ml_system:
                try:
                    # Combine all data for ML
                    ml_data = {
                        **asdict(contract_result),
                        **whale_activity,
                        **metrics
                    }
                    ml_score = self.ml_system.predict_risk(ml_data)
                except Exception as e:
                    logger.debug(f"ML prediction failed: {str(e)}")
            
            # 5. Calculate combined score
            contract_score = 1.0 - contract_result.risk_score
            whale_score = whale_activity.get('confidence_score', 0)
            liquidity_score = min(metrics.get('liquidity', 0) / 10000, 1.0)
            volume_score = min(metrics.get('volume_24h', 0) / 100000, 1.0)
            
            combined_score = (
                contract_score * 0.3 +
                whale_score * 0.3 +
                ml_score * 0.2 +
                liquidity_score * 0.1 +
                volume_score * 0.1
            )
            
            # 6. Check minimum thresholds
            is_opportunity = (
                combined_score >= self.config.min_confidence_score and
                contract_result.liquidity_amount >= self.config.min_liquidity and
                contract_result.sell_tax <= self.config.max_sell_tax and
                not contract_result.honeypot_detected
            )
            
            return {
                'token_address': token_address,
                'is_opportunity': is_opportunity,
                'combined_score': combined_score,
                'contract_score': contract_score,
                'whale_score': whale_score,
                'ml_score': ml_score,
                'liquidity_score': liquidity_score,
                'volume_score': volume_score,
                'contract_result': contract_result,
                'whale_activity': whale_activity,
                'metrics': metrics,
                'current_price': metrics.get('price', 0)
            }
        
        except Exception as e:
            logger.error(f"Analysis failed for {token_address}: {str(e)}")
            return {
                'token_address': token_address,
                'is_opportunity': False,
                'reason': f'Analysis error: {str(e)}'
            }
    
    async def _consider_trade(self, opportunity: Dict):
        """Consider executing a trade based on opportunity"""
        token_address = opportunity['token_address']
        
        # Check if already have position
        if token_address in self.portfolio.positions:
            logger.debug(f"Already have position in {token_address}")
            return
        
        # Check available balance
        available_balance = self.portfolio.get_available_balance()
        if available_balance < 0.001:  # Minimum $0.001
            logger.warning("Insufficient balance for trading")
            return
        
        # Calculate position size
        current_price = opportunity.get('current_price', 0)
        if current_price <= 0:
            logger.warning(f"Invalid price for {token_address}: {current_price}")
            return
        
        confidence = opportunity.get('combined_score', 0)
        position_size = self.portfolio.calculate_position_size(current_price, confidence, self.config)
        
        if position_size <= 0:
            logger.debug(f"Position size too small for {token_address}")
            return
        
        # Create position
        position = TradePosition(
            token_address=token_address,
            entry_price=current_price,
            quantity=position_size,
            entry_time=time.time(),
            stop_loss=current_price * (1 - self.config.stop_loss_pct),
            take_profit=current_price * (1 + self.config.take_profit_pct),
            confidence_score=confidence
        )
        
        # Execute trade
        success = await self._execute_buy_order(position)
        
        if success:
            self.portfolio.add_position(position)
            self.portfolio.daily_trades += 1
            logger.info(f"🎯 New position: {token_address[:8]} - Size: {position_size:.6f} - Price: ${current_price:.8f}")
    
    async def _execute_buy_order(self, position: TradePosition) -> bool:
        """Execute buy order"""
        if self.paper_trading:
            # Simulate trade execution
            logger.info(f"📝 PAPER: Buying {position.quantity:.6f} of {position.token_address[:8]} at ${position.entry_price:.8f}")
            position.transaction_hash = f"paper_trade_{int(time.time())}"
            return True
        
        try:
            # Real trading implementation would go here
            # This is a simplified version - real implementation needs:
            # 1. Jupiter swap API integration
            # 2. Transaction building and signing
            # 3. Error handling and retries
            # 4. Slippage protection
            
            logger.info(f"💰 Executing buy order for {position.token_address[:8]}")
            
            # For now, simulate successful execution
            await asyncio.sleep(1)  # Simulate network delay
            position.transaction_hash = f"simulated_{int(time.time())}"
            
            return True
        
        except Exception as e:
            logger.error(f"Buy order failed: {str(e)}")
            return False
    
    async def _monitor_positions(self):
        """Monitor existing positions for exit conditions"""
        if not self.portfolio.positions:
            return
        
        logger.debug(f"📊 Monitoring {len(self.portfolio.positions)} positions")
        
        # Get current prices
        token_addresses = list(self.portfolio.positions.keys())
        price_updates = await self._get_current_prices(token_addresses)
        
        # Update position prices
        self.portfolio.update_position_prices(price_updates)
        
        # Check exit conditions
        positions_to_close = []
        
        for token_address, position in self.portfolio.positions.items():
            current_price = price_updates.get(token_address, position.entry_price)
            
            # Check stop loss
            if current_price <= position.stop_loss:
                positions_to_close.append((token_address, current_price, "STOP_LOSS"))
            
            # Check take profit
            elif current_price >= position.take_profit:
                positions_to_close.append((token_address, current_price, "TAKE_PROFIT"))
            
            # Check time limit
            elif time.time() - position.entry_time > self.config.max_hold_time:
                positions_to_close.append((token_address, current_price, "TIME_LIMIT"))
        
        # Execute position closes
        for token_address, exit_price, reason in positions_to_close:
            await self._execute_sell_order(token_address, exit_price, reason)
    
    async def _get_current_prices(self, token_addresses: List[str]) -> Dict[str, float]:
        """Get current prices for tokens"""
        prices = {}
        
        for token_address in token_addresses:
            try:
                # Get price from Jupiter
                url = "https://quote-api.jup.ag/v6/quote"
                params = {
                    'inputMint': token_address,
                    'outputMint': 'So11111111111111111111111111111111111111112',
                    'amount': 1000000,  # 1 token
                    'slippageBps': 50
                }
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        out_amount = float(data.get('outAmount', 0))
                        if out_amount > 0:
                            prices[token_address] = out_amount / 1e9  # Convert to SOL
            
            except Exception as e:
                logger.debug(f"Price fetch failed for {token_address}: {str(e)}")
        
        return prices
    
    async def _execute_sell_order(self, token_address: str, exit_price: float, reason: str) -> bool:
        """Execute sell order"""
        if self.paper_trading:
            # Paper trading
            trade_result = self.portfolio.close_position(token_address, exit_price, reason)
            if trade_result:
                logger.info(f"📝 PAPER: Sold {token_address[:8]} - PnL: {trade_result.pnl_pct:.2%} - {reason}")
            return True
        
        try:
            # Real sell order implementation
            logger.info(f"💸 Executing sell order for {token_address[:8]} - {reason}")
            
            # Simulate sell execution
            await asyncio.sleep(1)
            
            trade_result = self.portfolio.close_position(token_address, exit_price, reason)
            return trade_result is not None
        
        except Exception as e:
            logger.error(f"Sell order failed: {str(e)}")
            return False
    
    async def _execute_trades(self):
        """Execute any pending trades"""
        # This would handle order queue, retries, etc.
        pass
    
    def _update_portfolio_metrics(self):
        """Update portfolio performance metrics"""
        summary = self.portfolio.get_portfolio_summary()
        
        # Log performance every 10 cycles
        if hasattr(self, '_cycle_count'):
            self._cycle_count += 1
        else:
            self._cycle_count = 1
        
        if self._cycle_count % 10 == 0:
            logger.info(
                f"💼 Portfolio: ${summary['current_balance']:.3f} "
                f"({summary['total_return']:+.1%}) - "
                f"{summary['total_trades']} trades - "
                f"{summary['win_rate']:.1%} win rate"
            )
    
    def stop_trading(self):
        """Stop the trading engine"""
        self.is_running = False
        logger.info("🛑 Trading engine stopped")
    
    async def emergency_stop(self):
        """Emergency stop - close all positions"""
        logger.warning("🚨 EMERGENCY STOP - Closing all positions")
        
        if not self.portfolio.positions:
            return
        
        # Get current prices
        token_addresses = list(self.portfolio.positions.keys())
        price_updates = await self._get_current_prices(token_addresses)
        
        # Close all positions
        for token_address in token_addresses:
            current_price = price_updates.get(token_address, 0)
            if current_price > 0:
                await self._execute_sell_order(token_address, current_price, "EMERGENCY")
        
        logger.info("🚨 Emergency stop completed")

def main():
    """Main CLI interface"""
    if len(sys.argv) < 2:
        print("Usage: python trading_engine.py [trade|paper|monitor|backtest|stop] [options]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    async def run_command():
        # Load configuration
        config = TradingConfig()
        
        # Get private key
        private_key = os.getenv('SOLANA_PRIVATE_KEY', '')
        if not private_key and command in ['trade', 'monitor']:
            print("❌ SOLANA_PRIVATE_KEY environment variable required")
            sys.exit(1)
        
        if command == "trade":
            # Live trading
            async with TradingEngine(private_key, config, paper_trading=False) as engine:
                await engine.start_trading()
        
        elif command == "paper":
            # Paper trading
            async with TradingEngine("", config, paper_trading=True) as engine:
                await engine.start_trading()
        
        elif command == "monitor":
            # Monitor only mode
            async with TradingEngine(private_key, config, paper_trading=True) as engine:
                engine.scan_interval = 60  # More frequent monitoring
                
                while True:
                    await engine._scan_for_opportunities()
                    await asyncio.sleep(60)
        
        elif command == "backtest":
            # Backtesting mode
            print("📊 Backtesting mode not implemented yet")
            # Would load historical data and simulate trading
        
        elif command == "stop":
            # Emergency stop
            print("🚨 Emergency stop not implemented for CLI")
            # Would need to communicate with running trading process
        
        elif command == "status":
            # Show current status
            try:
                db = TokenDatabase()
                recent_tokens = db.get_recent_tokens(1)
                print(f"📊 Recent tokens found: {len(recent_tokens)}")
                
                # Load portfolio if exists
                if os.path.exists('data/portfolio.pkl'):
                    with open('data/portfolio.pkl', 'rb') as f:
                        portfolio = pickle.load(f)
                    summary = portfolio.get_portfolio_summary()
                    print(f"💼 Portfolio balance: ${summary['current_balance']:.3f}")
                    print(f"📈 Total return: {summary['total_return']:+.1%}")
                    print(f"📊 Active positions: {summary['active_positions']}")
            except Exception as e:
                print(f"❌ Status check failed: {str(e)}")
        
        else:
            print(f"Unknown command: {command}")
            print("Available commands: trade, paper, monitor, backtest, stop, status")
    
    try:
        asyncio.run(run_command())
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
