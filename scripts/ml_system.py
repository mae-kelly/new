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
logging.basicConfig(
    level=logging.INFO,
    format='
    handlers=[
        logging.FileHandler('logs/ml_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
@dataclass
class TrainingConfig:
    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.risk_classifier = None
        self.volatility_predictor = None
        self.price_predictor = None
        self.feature_scaler = RobustScaler()
        self.target_scaler = StandardScaler()
        self.training_history = {
            'risk_classifier': [],
            'volatility_predictor': [],
            'price_predictor': []
        }
        logger.info(f"ML System initialized on device: {self.device}")
    def extract_features(self, token_data: Dict) -> np.ndarray:
        price_history = token_data.get('price_history', [])
        volume_history = token_data.get('volume_history', [])
        if len(price_history) < 20:
            return [0.0] * 30
        prices = np.array(price_history[-50:])
        volumes = np.array(volume_history[-50:]) if volume_history else np.ones(len(prices))
        features = []
        ma5 = self._moving_average(prices, 5)
        ma10 = self._moving_average(prices, 10)
        ma20 = self._moving_average(prices, 20)
        features.extend([
            (prices[-1] - ma5[-1]) / ma5[-1] if ma5[-1] > 0 else 0,
            (prices[-1] - ma10[-1]) / ma10[-1] if ma10[-1] > 0 else 0,
            (prices[-1] - ma20[-1]) / ma20[-1] if ma20[-1] > 0 else 0,
            (ma5[-1] - ma10[-1]) / ma10[-1] if ma10[-1] > 0 else 0,
            (ma10[-1] - ma20[-1]) / ma20[-1] if ma20[-1] > 0 else 0,
        ])
        rsi = self._calculate_rsi(prices, 14)
        features.append(rsi / 100.0)
        macd, signal = self._calculate_macd(prices)
        features.extend([
            macd / prices[-1] if prices[-1] > 0 else 0,
            signal / prices[-1] if prices[-1] > 0 else 0,
            (macd - signal) / prices[-1] if prices[-1] > 0 else 0
        ])
        bb_upper, bb_lower = self._calculate_bollinger_bands(prices, 20, 2)
        bb_position = (prices[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1]) if bb_upper[-1] > bb_lower[-1] else 0.5
        bb_width = (bb_upper[-1] - bb_lower[-1]) / ma20[-1] if ma20[-1] > 0 else 0
        features.extend([bb_position, bb_width])
        volume_ma = self._moving_average(volumes, 10)
        volume_ratio = volumes[-1] / volume_ma[-1] if volume_ma[-1] > 0 else 1
        features.append(min(volume_ratio, 10))
        momentum_1 = (prices[-1] - prices[-2]) / prices[-2] if len(prices) > 1 and prices[-2] > 0 else 0
        momentum_5 = (prices[-1] - prices[-6]) / prices[-6] if len(prices) > 5 and prices[-6] > 0 else 0
        momentum_10 = (prices[-1] - prices[-11]) / prices[-11] if len(prices) > 10 and prices[-11] > 0 else 0
        features.extend([momentum_1, momentum_5, momentum_10])
        returns = np.diff(np.log(prices + 1e-8))
        volatility = np.std(returns) if len(returns) > 0 else 0
        features.append(volatility)
        recent_high = np.max(prices[-10:]) if len(prices) >= 10 else prices[-1]
        recent_low = np.min(prices[-10:]) if len(prices) >= 10 else prices[-1]
        support_distance = (prices[-1] - recent_low) / recent_low if recent_low > 0 else 0
        resistance_distance = (recent_high - prices[-1]) / recent_high if recent_high > 0 else 0
        features.extend([support_distance, resistance_distance])
        atr = self._calculate_atr(prices, 14)
        features.append(atr / prices[-1] if prices[-1] > 0 else 0)
        stoch_k = self._calculate_stochastic(prices, 14)
        features.append(stoch_k / 100.0)
        williams_r = self._calculate_williams_r(prices, 14)
        features.append((williams_r + 100) / 100.0)
        cci = self._calculate_cci(prices, 20)
        features.append(max(-3, min(3, cci / 100.0)))
        roc = self._calculate_roc(prices, 10)
        features.append(roc / 100.0)
        while len(features) < 30:
            features.append(0.0)
        return features[:30]
    def _extract_sentiment_features(self, token_data: Dict) -> List[float]:
        features = []
        features.extend([
            token_data.get('ecosystem_score', 0),
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
        return features[:10] + [0.0] * max(0, 10 - len(features))
    def _moving_average(self, data: np.ndarray, window: int) -> np.ndarray:
