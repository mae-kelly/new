#!/usr/bin/env python3

import os
import sys
import json
import time
import pickle
import logging
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, mean_squared_error, accuracy_score
import optuna
from optuna.integration import PyTorchLightningPruningCallback

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ml_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for ML training"""
    model_type: str = "neural_net"  # neural_net, random_forest, gradient_boost
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.3
    hidden_dims: List[int] = None
    early_stopping_patience: int = 15
    use_mixup: bool = True
    use_label_smoothing: bool = True
    use_cosine_annealing: bool = True
    use_gradient_clipping: bool = True
    cross_validation_folds: int = 5
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128, 64]

class AdvancedMLSystem:
    """Advanced ML system with PhD-level optimizations"""
    
    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Models
        self.risk_classifier = None
        self.volatility_predictor = None
        self.price_predictor = None
        
        # Scalers
        self.feature_scaler = RobustScaler()
        self.target_scaler = StandardScaler()
        
        # Training history
        self.training_history = {
            'risk_classifier': [],
            'volatility_predictor': [],
            'price_predictor': []
        }
        
        logger.info(f"ML System initialized on device: {self.device}")
    
    def extract_features(self, token_data: Dict) -> np.ndarray:
        """Extract comprehensive ML features from token data"""
        features = []
        
        # Contract safety features (20 features)
        features.extend([
            float(token_data.get('honeypot_detected', 1)),
            token_data.get('sell_tax', 1.0),
            token_data.get('buy_tax', 0.0),
            float(token_data.get('liquidity_locked', 0)),
            np.log1p(token_data.get('liquidity_amount', 1)),  # Log transform
            token_data.get('risk_score', 1.0),
            float(token_data.get('ownership_renounced', 0)),
            np.log1p(token_data.get('holder_count', 1)),
            token_data.get('mint_authority_exists', 1),
            token_data.get('freeze_authority_exists', 1),
            # Additional safety metrics
            len(token_data.get('issues', [])),
            float(token_data.get('has_transfer_restrictions', 1)),
            float(token_data.get('has_freeze_authority', 1)),
            token_data.get('top_holder_percentage', 1.0),
            np.log1p(token_data.get('transaction_count_24h', 1)),
            token_data.get('age_hours', 0) / 24,  # Normalize to days
            float(token_data.get('verified', 0)),
            float(token_data.get('rugpull_risk', 1)),
            token_data.get('social_score', 0),
            token_data.get('dev_activity_score', 0)
        ])
        
        # Market data features (25 features)
        features.extend([
            np.log1p(token_data.get('price', 1e-10)),
            np.log1p(token_data.get('volume_24h', 1)),
            np.log1p(token_data.get('market_cap', 1)),
            token_data.get('price_change_1h', 0),
            token_data.get('price_change_24h', 0),
            token_data.get('volume_change_24h', 0),
            token_data.get('trades_24h', 0) / 1000,  # Normalize
            token_data.get('unique_traders_24h', 0) / 100,
            token_data.get('average_trade_size', 0),
            token_data.get('buy_sell_ratio', 0.5),
            # Volatility metrics
            token_data.get('volatility_1h', 0),
            token_data.get('volatility_24h', 0),
            token_data.get('volatility_7d', 0),
            # Liquidity metrics
            token_data.get('liquidity_depth_2pct', 0),
            token_data.get('liquidity_depth_5pct', 0),
            token_data.get('slippage_1k', 0),
            token_data.get('slippage_10k', 0),
            # Exchange metrics
            token_data.get('dex_count', 0),
            float(token_data.get('is_on_raydium', 0)),
            float(token_data.get('is_on_jupiter', 0)),
            float(token_data.get('is_on_orca', 0)),
            # Whale activity
            token_data.get('whale_transaction_count', 0),
            token_data.get('whale_volume_percentage', 0),
            np.log1p(token_data.get('large_transaction_count', 1)),
            token_data.get('insider_trading_score', 0)
        ])
        
        # Technical indicators (30 features)
        technical_features = self._calculate_technical_indicators(token_data)
        features.extend(technical_features)
        
        # Sentiment and social features (15 features)
        sentiment_features = self._extract_sentiment_features(token_data)
        features.extend(sentiment_features)
        
        # Network effect features (10 features)
        network_features = self._extract_network_features(token_data)
        features.extend(network_features)
        
        # Ensure fixed feature count (100 features total)
        target_features = 100
        if len(features) > target_features:
            features = features[:target_features]
        elif len(features) < target_features:
            features.extend([0.0] * (target_features - len(features)))
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_technical_indicators(self, token_data: Dict) -> List[float]:
        """Calculate technical analysis indicators"""
        price_history = token_data.get('price_history', [])
        volume_history = token_data.get('volume_history', [])
        
        if len(price_history) < 20:
            return [0.0] * 30
        
        prices = np.array(price_history[-50:])  # Last 50 periods
        volumes = np.array(volume_history[-50:]) if volume_history else np.ones(len(prices))
        
        features = []
        
        # Moving averages and crossovers
        ma5 = self._moving_average(prices, 5)
        ma10 = self._moving_average(prices, 10)
        ma20 = self._moving_average(prices, 20)
        
        features.extend([
            (prices[-1] - ma5[-1]) / ma5[-1] if ma5[-1] > 0 else 0,  # Price vs MA5
            (prices[-1] - ma10[-1]) / ma10[-1] if ma10[-1] > 0 else 0,  # Price vs MA10
            (prices[-1] - ma20[-1]) / ma20[-1] if ma20[-1] > 0 else 0,  # Price vs MA20
            (ma5[-1] - ma10[-1]) / ma10[-1] if ma10[-1] > 0 else 0,  # MA5 vs MA10
            (ma10[-1] - ma20[-1]) / ma20[-1] if ma20[-1] > 0 else 0,  # MA10 vs MA20
        ])
        
        # RSI
        rsi = self._calculate_rsi(prices, 14)
        features.append(rsi / 100.0)  # Normalize to 0-1
        
        # MACD
        macd, signal = self._calculate_macd(prices)
        features.extend([
            macd / prices[-1] if prices[-1] > 0 else 0,
            signal / prices[-1] if prices[-1] > 0 else 0,
            (macd - signal) / prices[-1] if prices[-1] > 0 else 0
        ])
        
        # Bollinger Bands
        bb_upper, bb_lower = self._calculate_bollinger_bands(prices, 20, 2)
        bb_position = (prices[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1]) if bb_upper[-1] > bb_lower[-1] else 0.5
        bb_width = (bb_upper[-1] - bb_lower[-1]) / ma20[-1] if ma20[-1] > 0 else 0
        features.extend([bb_position, bb_width])
        
        # Volume indicators
        volume_ma = self._moving_average(volumes, 10)
        volume_ratio = volumes[-1] / volume_ma[-1] if volume_ma[-1] > 0 else 1
        features.append(min(volume_ratio, 10))  # Cap at 10x
        
        # Price momentum
        momentum_1 = (prices[-1] - prices[-2]) / prices[-2] if len(prices) > 1 and prices[-2] > 0 else 0
        momentum_5 = (prices[-1] - prices[-6]) / prices[-6] if len(prices) > 5 and prices[-6] > 0 else 0
        momentum_10 = (prices[-1] - prices[-11]) / prices[-11] if len(prices) > 10 and prices[-11] > 0 else 0
        features.extend([momentum_1, momentum_5, momentum_10])
        
        # Volatility measures
        returns = np.diff(np.log(prices + 1e-8))
        volatility = np.std(returns) if len(returns) > 0 else 0
        features.append(volatility)
        
        # Support and resistance levels
        recent_high = np.max(prices[-10:]) if len(prices) >= 10 else prices[-1]
        recent_low = np.min(prices[-10:]) if len(prices) >= 10 else prices[-1]
        support_distance = (prices[-1] - recent_low) / recent_low if recent_low > 0 else 0
        resistance_distance = (recent_high - prices[-1]) / recent_high if recent_high > 0 else 0
        features.extend([support_distance, resistance_distance])
        
        # Average True Range (ATR)
        atr = self._calculate_atr(prices, 14)
        features.append(atr / prices[-1] if prices[-1] > 0 else 0)
        
        # Stochastic Oscillator
        stoch_k = self._calculate_stochastic(prices, 14)
        features.append(stoch_k / 100.0)
        
        # Williams %R
        williams_r = self._calculate_williams_r(prices, 14)
        features.append((williams_r + 100) / 100.0)  # Normalize to 0-1
        
        # Commodity Channel Index (CCI)
        cci = self._calculate_cci(prices, 20)
        features.append(max(-3, min(3, cci / 100.0)))  # Normalize and clip
        
        # Rate of Change (ROC)
        roc = self._calculate_roc(prices, 10)
        features.append(roc / 100.0)
        
        # Pad remaining features
        while len(features) < 30:
            features.append(0.0)
        
        return features[:30]
    
    def _extract_sentiment_features(self, token_data: Dict) -> List[float]:
        """Extract sentiment and social media features"""
        features = []
        
        # Social media metrics
        features.extend([
            token_data.get('twitter_followers', 0) / 1000,  # Normalized
            token_data.get('twitter_mentions_24h', 0) / 100,
            token_data.get('discord_members', 0) / 1000,
            token_data.get('telegram_members', 0) / 1000,
            token_data.get('reddit_subscribers', 0) / 1000,
            token_data.get('social_sentiment_score', 0),  # -1 to 1
            token_data.get('influencer_mentions', 0),
            token_data.get('news_mentions_24h', 0),
            token_data.get('youtube_videos_24h', 0),
            token_data.get('google_trends_score', 0),  # 0 to 100
            token_data.get('fear_greed_index', 50) / 100,  # Normalize to 0-1
            token_data.get('dev_activity_github', 0),
            token_data.get('community_activity_score', 0),
            token_data.get('hype_score', 0),
            token_data.get('fomo_indicator', 0)
        ])
        
        # Ensure exactly 15 features
        return features[:15] + [0.0] * max(0, 15 - len(features))
    
    def _extract_network_features(self, token_data: Dict) -> List[float]:
        """Extract network effect and ecosystem features"""
        features = []
        
        features.extend([
            token_data.get('ecosystem_score', 0),  # Solana ecosystem strength
            token_data.get('defi_integrations', 0),
            token_data.get('nft_connections', 0),
            token_data.get('bridge_availability', 0),
            token_data.get('staking_availability', 0),
            token_data.get('yield_farming_score', 0),
            token_data.get('partnership_score', 0),
            token_data.get('exchange_listings', 0),
            token_data.get('api_integrations', 0),
            token_data.get('network_effect_score', 0)
        ])
        
        # Ensure exactly 10 features
        return features[:10] + [0.0] * max(0, 10 - len(features))
    
    # Technical indicator calculation methods
    def _moving_average(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate moving average"""
        if len(data) < window:
            return np.array([data[-1]] * len(data))
        return np.convolve(data, np.ones(window)/window, mode='same')
    
    def _calculate
