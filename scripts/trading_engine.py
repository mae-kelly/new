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
import base58
from solana.rpc.async_api import AsyncClient
from solana.keypair import Keypair
from solders.pubkey import Pubkey
from solders.transaction import VersionedTransaction
from solders.instruction import Instruction
from solders.system_program import TransferParams, transfer
import aiohttp
sys.path.append('scripts')
from contract_analyzer import SolanaContractAnalyzer, ContractAnalysisResult
from token_discovery import TokenDiscovery, TokenDatabase
from ml_system import AdvancedMLSystem, TrainingEngine
logging.basicConfig(
    level=logging.INFO,
    format='
    handlers=[
        logging.FileHandler('logs/trading_engine.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
@dataclass
class TradingConfig:
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
    status: str = "OPEN"
@dataclass
class TradeResult:
    def __init__(self, initial_balance: float = 10.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.positions: Dict[str, TradePosition] = {}
        self.trade_history: List[TradeResult] = []
        self.daily_trades = 0
        self.last_trade_date = datetime.now().date()
    def get_available_balance(self) -> float:
        available = self.get_available_balance()
        max_position_value = available * config.max_position_size
        win_rate = self._estimate_win_rate()
        avg_win = self._estimate_avg_win()
        avg_loss = self._estimate_avg_loss()
        if avg_loss == 0:
            kelly_fraction = 0.1
        else:
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_fraction = max(0, min(kelly_fraction, 0.25))
        adjusted_fraction = kelly_fraction * confidence * config.risk_multiplier
        position_value = min(max_position_value, available * adjusted_fraction)
        return position_value / price if price > 0 else 0
    def _estimate_win_rate(self) -> float:
        wins = [trade.pnl_pct for trade in self.trade_history[-50:] if trade.pnl > 0]
        return np.mean(wins) if wins else 0.5
    def _estimate_avg_loss(self) -> float:
        if position.token_address in self.positions:
            logger.warning(f"Position already exists for {position.token_address}")
            return False
        self.positions[position.token_address] = position
        return True
    def close_position(self, token_address: str, exit_price: float, reason: str) -> Optional[TradeResult]:
        for token_address, current_price in price_updates.items():
            if token_address in self.positions:
                position = self.positions[token_address]
                position.current_price = current_price
                position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
    def get_portfolio_summary(self) -> Dict:
    def __init__(self, private_key: str, config: TradingConfig = None, paper_trading: bool = False):
        self.config = config or TradingConfig()
        self.paper_trading = paper_trading
        self.portfolio = PortfolioManager()
        self.session = None
        self.db = TokenDatabase()
        self.contract_analyzer = None
        self.token_discovery = None
        self.ml_system = None
        if not paper_trading:
            try:
                self.keypair = Keypair.from_secret_key(private_key)
                self.client = AsyncClient("https:
                logger.info(f"Trading wallet: {self.keypair.pubkey()}")
            except Exception as e:
                logger.error(f"Wallet setup failed: {str(e)}")
                raise
        else:
            self.keypair = None
            self.client = None
            logger.info("Paper trading mode enabled")
        self.is_running = False
        self.last_scan_time = 0
        self.scan_interval = 300
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        self.contract_analyzer = await SolanaContractAnalyzer().__aenter__()
        self.token_discovery = await TokenDiscovery().__aenter__()
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
        try:
            await self._scan_for_opportunities()
            await self._monitor_positions()
            await self._execute_trades()
            self._update_portfolio_metrics()
        except Exception as e:
            logger.error(f"Trading cycle error: {str(e)}")
    async def _scan_for_opportunities(self):
        try:
            contract_result = await self.contract_analyzer.analyze_token(token_address)
            if not contract_result.is_safe:
                return {
                    'token_address': token_address,
                    'is_opportunity': False,
                    'reason': 'Contract unsafe',
                    'contract_score': 0
                }
            whale_signals = await self.token_discovery.track_whale_activity([token_address])
            whale_activity = whale_signals.get(token_address, {})
            metrics = await self.token_discovery.get_token_detailed_metrics(token_address)
            ml_score = 0.5
            if self.ml_system:
                try:
                    ml_data = {
                    }
                    ml_score = self.ml_system.predict_risk(ml_data)
                except Exception as e:
                    logger.debug(f"ML prediction failed: {str(e)}")
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
        if self.paper_trading:
            logger.info(f"📝 PAPER: Buying {position.quantity:.6f} of {position.token_address[:8]} at ${position.entry_price:.8f}")
            position.transaction_hash = f"paper_trade_{int(time.time())}"
            return True
        try:
            logger.info(f"💰 Executing buy order for {position.token_address[:8]}")
            await asyncio.sleep(1)
            position.transaction_hash = f"simulated_{int(time.time())}"
            return True
        except Exception as e:
            logger.error(f"Buy order failed: {str(e)}")
            return False
    async def _monitor_positions(self):
        prices = {}
        for token_address in token_addresses:
            try:
                url = "https:
                params = {
                    'inputMint': token_address,
                    'outputMint': 'So11111111111111111111111111111111111111112',
                    'amount': 1000000,
                    'slippageBps': 50
                }
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        out_amount = float(data.get('outAmount', 0))
                        if out_amount > 0:
                            prices[token_address] = out_amount / 1e9
            except Exception as e:
                logger.debug(f"Price fetch failed for {token_address}: {str(e)}")
        return prices
    async def _execute_sell_order(self, token_address: str, exit_price: float, reason: str) -> bool:
        pass
    def _update_portfolio_metrics(self):
        self.is_running = False
        logger.info("🛑 Trading engine stopped")
    async def emergency_stop(self):
    if len(sys.argv) < 2:
        print("Usage: python trading_engine.py [trade|paper|monitor|backtest|stop] [options]")
        sys.exit(1)
    command = sys.argv[1]
    async def run_command():
        config = TradingConfig()
        private_key = os.getenv('SOLANA_PRIVATE_KEY', '')
        if not private_key and command in ['trade', 'monitor']:
            print("❌ SOLANA_PRIVATE_KEY environment variable required")
            sys.exit(1)
        if command == "trade":
            async with TradingEngine(private_key, config, paper_trading=False) as engine:
                await engine.start_trading()
        elif command == "paper":
            async with TradingEngine("", config, paper_trading=True) as engine:
                await engine.start_trading()
        elif command == "monitor":
            async with TradingEngine(private_key, config, paper_trading=True) as engine:
                engine.scan_interval = 60
                while True:
                    await engine._scan_for_opportunities()
                    await asyncio.sleep(60)
        elif command == "backtest":
            print("📊 Backtesting mode not implemented yet")
        elif command == "stop":
            print("🚨 Emergency stop not implemented for CLI")
        elif command == "status":
            try:
                db = TokenDatabase()
                recent_tokens = db.get_recent_tokens(1)
                print(f"📊 Recent tokens found: {len(recent_tokens)}")
                if os.path.exists('data/portfolio.pkl'):
                    with open('data/portfolio.pkl', 'rb') as f:
                        portfolio = pickle.load(f)
                    summary = portfolio.get_portfolio_summary()
                    print(f"💼 Portfolio balance: ${summary['current_balance']:.3f}")
                    print(f"📈 Total return: {summary['total_return']:+.1
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
