import numpy as np
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass
@dataclass
class RiskMetrics:
    def __init__(self):
        self.max_correlation = 0.7
        self.max_concentration = 0.3
        self.var_confidence = 0.05
    def calculate_portfolio_risk(self, positions: Dict, returns_history: List[float]) -> RiskMetrics:
