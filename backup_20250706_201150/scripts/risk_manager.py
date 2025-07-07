#!/usr/bin/env python3

import numpy as np
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class RiskMetrics:
    """Portfolio risk metrics"""
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    value_at_risk: float
    expected_shortfall: float
    beta: float
    alpha: float

class RiskManager:
    """Advanced risk management system"""
    
    def __init__(self):
        self.max_correlation = 0.7
        self.max_concentration = 0.3
        self.var_confidence = 0.05  # 95% VaR
    
    def calculate_portfolio_risk(self, positions: Dict, returns_history: List[float]) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        if len(returns_history) < 30:
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0)
        
        returns = np.array(returns_history)
        
        # Maximum drawdown
        cumulative = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Sharpe ratio
        risk_free_rate = 0.02 / 365  # Daily risk-free rate
        excess_returns = returns - risk_free_rate
        sharpe_ratio = np.mean(excess_returns) / np.std(returns) * np.sqrt(365) if np.std(returns) > 0 else 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0
