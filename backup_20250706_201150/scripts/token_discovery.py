#!/usr/bin/env python3

import asyncio
import aiohttp
import json
import time
import logging
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, asdict
import sqlite3
import base58
from solana.rpc.async_api import AsyncClient
from solders.pubkey import Pubkey
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/token_discovery.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TokenInfo:
    """Token information structure"""
    address: str
    name: str = ""
    symbol: str = ""
    created_timestamp: float = 0
    price: float = 0
    volume_24h: float = 0
    market_cap: float = 0
    liquidity: float = 0
    holder_count: int = 0
    source: str = ""  # pump.fun, raydium, etc.
    discovery_time: float = 0
    price_change_1h: float = 0
    price_change_24h: float = 0
    is_analyzed: bool = False
    risk_score: float = 1.0
    whale_activity: Dict = None
    
    def __post_init__(self):
        if self.discovery_time == 0:
            self.discovery_time = time.time()
        if self.whale_activity is None:
            self.whale_activity = {}

@dataclass
class WhaleWallet:
    """Whale wallet tracking"""
    address: str
    total_profit: float = 0
    win_rate: float = 0
    avg_hold_time: float = 0
    recent_trades: List[Dict] = None
    confidence_score: float = 0
    last_activity: float = 0
    
    def __post_init__(self):
        if self.recent_trades is None:
            self.recent_trades = []

class TokenDatabase:
    """SQLite database for token tracking"""
    
    def __init__(self, db_path: str = "data/tokens.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tokens (
                    address TEXT PRIMARY KEY,
                    name TEXT,
                    symbol TEXT,
                    created_timestamp REAL,
                    price REAL,
                    volume_24h REAL,
                    market_cap REAL,
                    liquidity REAL,
                    holder_count INTEGER,
                    source TEXT,
                    discovery_time REAL,
                    price_change_1h REAL,
                    price_change_24h REAL,
                    is_analyzed BOOLEAN,
                    risk_score REAL,
                    whale_activity TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS whale_wallets (
                    address TEXT PRIMARY KEY,
                    total_profit REAL,
                    win_rate REAL,
                    avg_hold_time REAL,
                    confidence_score REAL,
                    last_activity REAL,
                    recent_trades TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS price_history (
                    token_address TEXT,
                    timestamp REAL,
                    price REAL,
                    volume REAL,
                    PRIMARY KEY (token_address, timestamp)
                )
            """)
            
            conn.commit()
    
    def save_token(self, token: TokenInfo):
        """Save token to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO tokens VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                token.address, token.name, token.symbol, token.created_timestamp,
                token.price, token.volume_24h, token.market_cap, token.liquidity,
                token.holder_count, token.source, token.discovery_time,
                token.price_change_1h, token.price_change_24h, token.is_analyzed,
                token.risk_score, json.dumps(token.whale_activity)
            ))
            conn.commit()
    
    def get_token(self, address: str) -> Optional[TokenInfo]:
        """Get token from database"""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT * FROM tokens WHERE address = ?", (address,)).fetchone()
            if row:
                whale_activity = json.loads(row[15]) if row[15] else {}
                return TokenInfo(
                    address=row[0], name=row[1], symbol=row[2], created_timestamp=row[3],
                    price=row[4], volume_24h=row[5], market_cap=row[6], liquidity=row[7],
                    holder_count=row[8], source=row[9], discovery_time=row[10],
                    price_change_1h=row[11], price_change_24h=row[12], is_analyzed=row[13],
                    risk_score=row[14], whale_activity=whale_activity
                )
        return None
    
    def get_recent_tokens(self, hours: int = 24) -> List[TokenInfo]:
        """Get tokens discovered in the last N hours"""
        cutoff = time.time() - (hours * 3600)
        tokens = []
        
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM tokens WHERE discovery_time > ? ORDER BY discovery_time DESC", 
                (cutoff,)
            ).fetchall()
            
            for row in rows:
                whale_activity = json.loads(row[15]) if row[15] else {}
                tokens.append(TokenInfo(
                    address=row[0], name=row[1], symbol=row[2], created_timestamp=row[3],
                    price=row[4], volume_24h=row[5], market_cap=row[6], liquidity=row[7],
                    holder_count=row[8], source=row[9], discovery_time=row[10],
                    price_change_1h=row[11], price_change_24h=row[12], is_analyzed=row[13],
                    risk_score=row[14], whale_activity=whale_activity
                ))
        
        return tokens
    
    def save_whale_wallet(self, whale: WhaleWallet):
        """Save whale wallet to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO whale_wallets VALUES (?,?,?,?,?,?,?)
            """, (
                whale.address, whale.total_profit, whale.win_rate, whale.avg_hold_time,
                whale.confidence_score, whale.last_activity, json.dumps(whale.recent_trades)
            ))
            conn.commit()
    
    def get_whale_wallets(self) -> List[WhaleWallet]:
        """Get all whale wallets"""
        whales = []
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT * FROM whale_wallets").fetchall()
            for row in rows:
                recent_trades = json.loads(row[6]) if row[6] else []
                whales.append(WhaleWallet(
                    address=row[0], total_profit=row[1], win_rate=row[2],
                    avg_hold_time=row[3], confidence_score=row[4], last_activity=row[5],
                    recent_trades=recent_trades
                ))
        return whales
    
    def save_price_point(self, token_address: str, price: float, volume: float):
        """Save price history point"""
        timestamp = time.time()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO price_history VALUES (?,?,?,?)
            """, (token_address, timestamp, price, volume))
            conn.commit()

class TokenDiscovery:
    """Advanced token discovery system"""
    
    def __init__(self):
        self.db = TokenDatabase()
        self.session = None
        self.solana_client = AsyncClient("https://api.mainnet-beta.solana.com")
        self.monitored_tokens: Set[str] = set()
        self.whale_wallets = self._load_whale_wallets()
        self.discovery_stats = {
            'tokens_found': 0,
            'whales_detected': 0,
            'profitable_signals': 0,
            'scan_cycles': 0
        }
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=100)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
        await self.solana_client.close()
    
    def _load_whale_wallets(self) -> List[WhaleWallet]:
        """Load known whale wallets"""
        whales = self.db.get_whale_wallets()
        
        # Add some known profitable wallets if database is empty
        if not whales:
            default_whales = [
                WhaleWallet("7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU", confidence_score=0.8),
                WhaleWallet("GDfnEsia2WLAW5t8yx2X5j2mkfA74i5kwGdDuZHt7XmG", confidence_score=0.7),
                WhaleWallet("BXP2gNKuqsmSKFRWqQKUBzHgrw2DG6QXd3JYHL5L6QqZ", confidence_score=0.6),
            ]
            for whale in default_whales:
                self.db.save_whale_wallet(whale)
            whales = default_whales
        
        return whales
    
    async def scan_new_tokens(self, max_age_minutes: int = 30) -> List[TokenInfo]:
        """Scan for new tokens across multiple sources"""
        logger.info(f"🔍 Scanning for tokens newer than {max_age_minutes} minutes...")
        
        cutoff_time = time.time() - (max_age_minutes * 60)
        new_tokens = []
        
        # Scan multiple sources in parallel
        tasks = [
            self._scan_pump_fun(cutoff_time),
            self._scan_raydium(cutoff_time),
            self._scan_jupiter_new_listings(cutoff_time),
            self._scan_dexscreener(cutoff_time)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        for result in results:
            if isinstance(result, list):
                new_tokens.extend(result)
            elif isinstance(result, Exception):
                logger.warning(f"Scan failed: {str(result)}")
        
        # Remove duplicates and filter
        unique_tokens = {}
        for token in new_tokens:
            if token.address not in unique_tokens:
                unique_tokens[token.address] = token
        
        filtered_tokens = list(unique_tokens.values())
        
        # Save to database
        for token in filtered_tokens:
            self.db.save_token(token)
        
        self.discovery_stats['tokens_found'] += len(filtered_tokens)
        self.discovery_stats['scan_cycles'] += 1
        
        logger.info(f"✅ Found {len(filtered_tokens)} new tokens")
        return filtered_tokens
    
    async def _scan_pump_fun(self, cutoff_time: float) -> List[TokenInfo]:
        """Scan pump.fun for new token launches"""
        tokens = []
        try:
            url = "https://frontend-api.pump.fun/coins"
            params = {
                'offset': 0,
                'limit': 100,
                'sort': 'created_timestamp',
                'order': 'DESC'
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    for coin in data:
                        created_time = coin.get('created_timestamp', 0)
                        if created_time > cutoff_time:
                            token = TokenInfo(
                                address=coin.get('mint', ''),
                                name=coin.get('name', ''),
                                symbol=coin.get('symbol', ''),
                                created_timestamp=created_time,
                                market_cap=coin.get('market_cap', 0),
                                volume_24h=coin.get('volume_24h', 0),
                                source="pump.fun",
                                price=coin.get('price_usd', 0)
                            )
                            if token.address:
                                tokens.append(token)
                else:
                    logger.warning(f"pump.fun API error: {response.status}")
        
        except Exception as e:
            logger.error(f"pump.fun scan error: {str(e)}")
        
        return tokens
    
    async def _scan_raydium(self, cutoff_time: float) -> List[TokenInfo]:
        """Scan Raydium for new pairs"""
        tokens = []
        try:
            url = "https://api.raydium.io/v2/main/pairs"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    for pair in data:
                        created_at = pair.get('created_at', 0)
                        if created_at > cutoff_time:
                            base_mint = pair.get('baseMint', '')
                            if base_mint and base_mint != 'So11111111111111111111111111111111111111112':
                                token = TokenInfo(
                                    address=base_mint,
                                    name=pair.get('baseToken', {}).get('name', ''),
                                    symbol=pair.get('baseToken', {}).get('symbol', ''),
                                    created_timestamp=created_at,
                                    liquidity=pair.get('liquidity', 0),
                                    volume_24h=pair.get('volume24h', 0),
                                    source="raydium",
                                    price=pair.get('price', 0)
                                )
                                tokens.append(token)
                else:
                    logger.warning(f"Raydium API error: {response.status}")
        
        except Exception as e:
            logger.error(f"Raydium scan error: {str(e)}")
        
        return tokens
    
    async def _scan_jupiter_new_listings(self, cutoff_time: float) -> List[TokenInfo]:
        """Scan Jupiter for new token listings"""
        tokens = []
        try:
            # Get all tokens from Jupiter
            url = "https://quote-api.jup.ag/v6/tokens"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Check for newly added tokens (simplified)
                    for token_addr, token_info in data.items():
                        if token_addr != 'So11111111111111111111111111111111111111112':
                            # Check if this is a new discovery
                            existing_token = self.db.get_token(token_addr)
                            if not existing_token:
                                token = TokenInfo(
                                    address=token_addr,
                                    name=token_info.get('name', ''),
                                    symbol=token_info.get('symbol', ''),
                                    source="jupiter",
                                    created_timestamp=time.time()  # Approximate
                                )
                                tokens.append(token)
                                
                                # Limit to prevent overwhelming
                                if len(tokens) >= 20:
                                    break
        
        except Exception as e:
            logger.error(f"Jupiter scan error: {str(e)}")
        
        return tokens
    
    async def _scan_dexscreener(self, cutoff_time: float) -> List[TokenInfo]:
        """Scan DexScreener for new Solana tokens"""
        tokens = []
        try:
            url = "https://api.dexscreener.com/latest/dex/tokens/solana"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    for pair in data.get('pairs', []):
                        pair_created = pair.get('pairCreatedAt', 0)
                        if pair_created and pair_created > cutoff_time * 1000:  # DexScreener uses milliseconds
                            base_token = pair.get('baseToken', {})
                            if base_token.get('address'):
                                token = TokenInfo(
                                    address=base_token['address'],
                                    name=base_token.get('name', ''),
                                    symbol=base_token.get('symbol', ''),
                                    created_timestamp=pair_created / 1000,
                                    price=float(pair.get('priceUsd', 0)),
                                    volume_24h=float(pair.get('volume', {}).get('h24', 0)),
                                    liquidity=float(pair.get('liquidity', {}).get('usd', 0)),
                                    source="dexscreener",
                                    price_change_1h=float(pair.get('priceChange', {}).get('h1', 0)),
                                    price_change_24h=float(pair.get('priceChange', {}).get('h24', 0))
                                )
                                tokens.append(token)
        
        except Exception as e:
            logger.error(f"DexScreener scan error: {str(e)}")
        
        return tokens
    
    async def track_whale_activity(self, token_addresses: List[str]) -> Dict[str, Dict]:
        """Track whale wallet activity for given tokens"""
        logger.info(f"🐋 Tracking whale activity for {len(token_addresses)} tokens...")
        
        whale_signals = {}
        
        for token_address in token_addresses:
            whale_activity = {
                'whale_buyers': [],
                'whale_volume': 0,
                'confidence_score': 0,
                'signal_strength': 0
            }
            
            # Check each whale wallet
            for whale in self.whale_wallets:
                try:
                    activity = await self._check_wallet_token_activity(whale.address, token_address)
                    
                    if activity['has_bought']:
                        whale_activity['whale_buyers'].append({
                            'wallet': whale.address,
                            'confidence': whale.confidence_score,
                            'volume': activity['volume'],
                            'timestamp': activity['timestamp']
                        })
                        whale_activity['whale_volume'] += activity['volume']
                        
                        self.discovery_stats['whales_detected'] += 1
                
                except Exception as e:
                    logger.warning(f"Whale check failed for {whale.address}: {str(e)}")
            
            # Calculate confidence score
            if whale_activity['whale_buyers']:
                # Weight by whale confidence and volume
                total_confidence = sum(buyer['confidence'] for buyer in whale_activity['whale_buyers'])
                volume_factor = min(whale_activity['whale_volume'] / 10000, 2.0)  # Cap at 2x
                recency_factor = self._calculate_recency_factor(whale_activity['whale_buyers'])
                
                whale_activity['confidence_score'] = min(
                    (total_confidence * volume_factor * recency_factor) / 10,
                    1.0
                )
                
                whale_activity['signal_strength'] = len(whale_activity['whale_buyers'])
                
                if whale_activity['confidence_score'] > 0.7:
                    self.discovery_stats['profitable_signals'] += 1
            
            whale_signals[token_address] = whale_activity
        
        return whale_signals
    
    async def _check_wallet_token_activity(self, wallet_address: str, token_address: str) -> Dict:
        """Check if wallet has recent activity with specific token"""
        try:
            pubkey = Pubkey.from_string(wallet_address)
            
            # Get recent transactions
            recent_txs = await self.solana_client.get_signatures_for_address(
                pubkey, limit=50
            )
            
            volume = 0
            has_bought = False
            latest_timestamp = 0
            
            for tx_info in recent_txs.value:
                # Check if transaction is recent (last 24 hours)
                if time.time() - tx_info.block_time > 86400:
                    continue
                
                try:
                    # Get transaction details
                    tx_detail = await self.solana_client.get_transaction(
                        tx_info.signature,
                        encoding="json",
                        max_supported_transaction_version=0
                    )
                    
                    if tx_detail.value:
                        # Simplified check for token involvement
                        tx_str = str(tx_detail.value)
                        if token_address in tx_str:
                            has_bought = True
                            volume += 1000  # Simplified volume calculation
                            latest_timestamp = max(latest_timestamp, tx_info.block_time)
                
                except Exception as e:
                    logger.debug(f"Transaction detail fetch failed: {str(e)}")
                    continue
            
            return {
                'has_bought': has_bought,
                'volume': volume,
                'timestamp': latest_timestamp
            }
        
        except Exception as e:
            logger.warning(f"Wallet activity check failed: {str(e)}")
            return {'has_bought': False, 'volume': 0, 'timestamp': 0}
    
    def _calculate_recency_factor(self, whale_buyers: List[Dict]) -> float:
        """Calculate recency factor based on when whales bought"""
        if not whale_buyers:
            return 0
        
        current_time = time.time()
        recency_scores = []
        
        for buyer in whale_buyers:
            time_diff = current_time - buyer.get('timestamp', current_time)
            # More recent = higher score (exponential decay)
            recency_score = np.exp(-time_diff / 3600)  # 1 hour decay constant
            recency_scores.append(recency_score)
        
        return np.mean(recency_scores)
    
    async def get_token_detailed_metrics(self, token_address: str) -> Dict:
        """Get comprehensive token metrics"""
        metrics = {
            '
