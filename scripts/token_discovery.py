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
logging.basicConfig(
    level=logging.INFO,
    format='
    handlers=[
        logging.FileHandler('logs/token_discovery.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
@dataclass
class TokenInfo:
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
        with sqlite3.connect(self.db_path) as conn:
            conn.commit()
    def save_token(self, token: TokenInfo):
                INSERT OR REPLACE INTO tokens VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
        with sqlite3.connect(self.db_path) as conn:
                whale.address, whale.total_profit, whale.win_rate, whale.avg_hold_time,
                whale.confidence_score, whale.last_activity, json.dumps(whale.recent_trades)
            ))
            conn.commit()
    def get_whale_wallets(self) -> List[WhaleWallet]:
        timestamp = time.time()
        with sqlite3.connect(self.db_path) as conn:
            conn.commit()
class TokenDiscovery:
        whales = self.db.get_whale_wallets()
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
        tokens = []
        try:
            url = "https:
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
        tokens = []
        try:
            url = "https:
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    for token_addr, token_info in data.items():
                        if token_addr != 'So11111111111111111111111111111111111111112':
                            existing_token = self.db.get_token(token_addr)
                            if not existing_token:
                                token = TokenInfo(
                                    address=token_addr,
                                    name=token_info.get('name', ''),
                                    symbol=token_info.get('symbol', ''),
                                    source="jupiter",
                                    created_timestamp=time.time()
                                )
                                tokens.append(token)
                                if len(tokens) >= 20:
                                    break
        except Exception as e:
            logger.error(f"Jupiter scan error: {str(e)}")
        return tokens
    async def _scan_dexscreener(self, cutoff_time: float) -> List[TokenInfo]:
        logger.info(f"🐋 Tracking whale activity for {len(token_addresses)} tokens...")
        whale_signals = {}
        for token_address in token_addresses:
            whale_activity = {
                'whale_buyers': [],
                'whale_volume': 0,
                'confidence_score': 0,
                'signal_strength': 0
            }
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
            if whale_activity['whale_buyers']:
                total_confidence = sum(buyer['confidence'] for buyer in whale_activity['whale_buyers'])
                volume_factor = min(whale_activity['whale_volume'] / 10000, 2.0)
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
        if not whale_buyers:
            return 0
        current_time = time.time()
        recency_scores = []
        for buyer in whale_buyers:
            time_diff = current_time - buyer.get('timestamp', current_time)
            recency_score = np.exp(-time_diff / 3600)
            recency_scores.append(recency_score)
        return np.mean(recency_scores)
    async def get_token_detailed_metrics(self, token_address: str) -> Dict:
