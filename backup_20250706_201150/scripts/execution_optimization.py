#!/usr/bin/env python3

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import cvxpy as cp
from scipy.optimize import minimize
from dataclasses import dataclass
import asyncio
import time

@dataclass
class ExecutionStrategy:
    """Optimal execution strategy"""
    total_quantity: float
    time_horizon: float  # seconds
    slice_sizes: List[float]
    slice_times: List[float]
    expected_cost: float
    risk_measure: float

class OptimalExecutionEngine:
    """Advanced execution optimization using Almgren-Chriss and TWAP/VWAP strategies"""
    
    def __init__(self):
        self.risk_aversion = 1e-6  # Risk aversion parameter
        self.volatility = 0.02     # Daily volatility
        self.temp_impact_coeff = 1e-6  # Temporary impact coefficient
        self.perm_impact_coeff = 1e-7  # Permanent impact coefficient
    
    def almgren_chriss_strategy(self, total_quantity: float, 
                               time_horizon: float,
                               n_slices: int = 10) -> ExecutionStrategy:
        """Almgren-Chriss optimal execution strategy"""
        
        # Time discretization
        dt = time_horizon / n_slices
        times = np.linspace(0, time_horizon, n_slices + 1)
        
        # Set up optimization problem
        x = cp.Variable(n_slices + 1)  # Holdings at each time
        u = cp.Variable(n_slices)      # Trade rates
        
        # Constraints
        constraints = [
            x[0] == total_quantity,    # Initial holding
            x[-1] == 0,                # Final holding (fully executed)
        ]
        
        # Holdings evolution: x[t+1] = x[t] - u[t] * dt
        for t in range(n_slices):
            constraints.append(x[t+1] == x[t] - u[t] * dt)
        
        # Cost components
        permanent_cost = 0
        temporary_cost = 0
        risk_penalty = 0
        
        for t in range(n_slices):
            # Permanent impact cost
            permanent_cost += self.perm_impact_coeff * u[t] * dt
            
            # Temporary impact cost
            temporary_cost += self.temp_impact_coeff * cp.abs(u[t]) * u[t] * dt
            
            # Risk penalty (variance of remaining position)
            risk_penalty += 0.5 * self.risk_aversion * (self.volatility**2) * x[t]**2 * dt
        
        # Objective: minimize expected cost + risk penalty
        objective = cp.Minimize(permanent_cost + temporary_cost + risk_penalty)
        
        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if x.value is None:
            # Fallback to uniform execution
            slice_sizes = [total_quantity / n_slices] * n_slices
            slice_times = times[1:].tolist()
        else:
            slice_sizes = (u.value * dt).tolist()
            slice_times = times[1:].tolist()
        
        expected_cost = problem.value if problem.value else 0.01 * total_quantity
        
        return ExecutionStrategy(
            total_quantity=total_quantity,
            time_horizon=time_horizon,
            slice_sizes=slice_sizes,
            slice_times=slice_times,
            expected_cost=expected_cost,
            risk_measure=np.std(slice_sizes) if slice_sizes else 0
        )
    
    def vwap_strategy(self, total_quantity: float,
                     historical_volume: List[float],
                     time_horizon: float) -> ExecutionStrategy:
        """Volume Weighted Average Price execution strategy"""
        
        if not historical_volume:
            # Fallback to uniform if no volume data
            n_slices = 10
            slice_sizes = [total_quantity / n_slices] * n_slices
            slice_times = np.linspace(time_horizon/n_slices, time_horizon, n_slices).tolist()
        else:
            # Allocate trades proportional to historical volume
            total_volume = sum(historical_volume)
            volume_weights = [v / total_volume for v in historical_volume]
            
            slice_sizes = [total_quantity * w for w in volume_weights]
            n_slices = len(slice_sizes)
            slice_times = np.linspace(time_horizon/n_slices, time_horizon, n_slices).tolist()
        
        # Estimate cost (simplified)
        expected_cost = sum(size * self.temp_impact_coeff * np.sqrt(size) for size in slice_sizes)
        
        return ExecutionStrategy(
            total_quantity=total_quantity,
            time_horizon=time_horizon,
            slice_sizes=slice_sizes,
            slice_times=slice_times,
            expected_cost=expected_cost,
            risk_measure=np.std(slice_sizes)
        )
    
    def implementation_shortfall_strategy(self, total_quantity: float,
                                        time_horizon: float,
                                        price_drift: float = 0.0) -> ExecutionStrategy:
        """Implementation Shortfall minimization strategy"""
        
        # Account for price drift (alpha)
        adjusted_perm_impact = self.perm_impact_coeff + price_drift
        
        # Optimal trading rate (constant for IS strategy)
        kappa = np.sqrt(self.risk_aversion * self.volatility**2 / self.temp_impact_coeff)
        
        if kappa * time_horizon < 1e-6:
            # Almost immediate execution
            n_slices = 1
            slice_sizes = [total_quantity]
            slice_times = [time_horizon]
        else:
            # Exponential decay profile
            n_slices = max(10, int(time_horizon / 60))  # At least 10 slices, 1 per minute
            times = np.linspace(0, time_horizon, n_slices + 1)
            
            # Holdings trajectory: exponential decay
            holdings = [total_quantity * np.exp(-kappa * t) for t in times]
            
            # Trade sizes
            slice_sizes = [holdings[i] - holdings[i+1] for i in range(n_slices)]
            slice_times = times[1:].tolist()
        
        # Expected cost
        expected_cost = (adjusted_perm_impact * total_quantity + 
                        0.5 * self.temp_impact_coeff * kappa * total_quantity**2)
        
        return ExecutionStrategy(
            total_quantity=total_quantity,
            time_horizon=time_horizon,
            slice_sizes=slice_sizes,
            slice_times=slice_times,
            expected_cost=expected_cost,
            risk_measure=kappa
        )
    
    def adaptive_strategy(self, total_quantity: float,
                         time_horizon: float,
                         market_conditions: Dict) -> ExecutionStrategy:
        """Adaptive strategy that switches based on market conditions"""
        
        volatility = market_conditions.get('volatility', self.volatility)
        volume = market_conditions.get('avg_volume', 1000000)
        spread = market_conditions.get('spread', 0.001)
        
        # Adjust parameters based on market conditions
        self.volatility = volatility
        self.temp_impact_coeff = spread / np.sqrt(volume)
        
        # Choose strategy based on conditions
        if volatility > 0.05:  # High volatility
            # Use faster execution to reduce risk
            return self.implementation_shortfall_strategy(total_quantity, time_horizon * 0.5)
        elif volume < 100000:  # Low volume
            # Use slower, smaller slices
            return self.vwap_strategy(total_quantity, [], time_horizon * 2)
        else:  # Normal conditions
            return self.almgren_chriss_strategy(total_quantity, time_horizon)
    
    async def execute_strategy(self, strategy: ExecutionStrategy,
                             execute_slice_func) -> Dict:
        """Execute the optimal strategy"""
        executed_quantities = []
        execution_prices = []
        total_cost = 0
        
        for i, (size, target_time) in enumerate(zip(strategy.slice_sizes, strategy.slice_times)):
            if size <= 0:
                continue
            
            # Wait until target time
            if i > 0:
                elapsed = time.time() - start_time
                sleep_time = target_time - elapsed
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
            else:
                start_time = time.time()
            
            # Execute slice
            try:
                result = await execute_slice_func(size)
                executed_quantities.append(result['quantity'])
                execution_prices.append(result['price'])
                total_cost += result['cost']
                
                print(f"Executed slice {i+1}: {result['quantity']:.2f} @ ${result['price']:.6f}")
                
            except Exception as e:
                print(f"Slice {i+1} execution failed: {str(e)}")
                executed_quantities.append(0)
                execution_prices.append(0)
        
        return {
            'total_executed': sum(executed_quantities),
            'total_cost': total_cost,
            'avg_price': np.mean([p for p in execution_prices if p > 0]),
            'execution_times': strategy.slice_times,
            'slippage': total_cost - strategy.expected_cost
        }

def main():
    """Test execution optimization strategies"""
    print("🎯 Testing Execution Optimization Strategies...")
    
    optimizer = OptimalExecutionEngine()
    
    # Test parameters
    total_quantity = 10000
    time_horizon = 3600  # 1 hour
    
    # Test Almgren-Chriss strategy
    ac_strategy = optimizer.almgren_chriss_strategy(total_quantity, time_horizon)
    print(f"✅ Almgren-Chriss: {len(ac_strategy.slice_sizes)} slices")
    print(f"   Expected cost: {ac_strategy.expected_cost:.2f}")
    print(f"   Risk measure: {ac_strategy.risk_measure:.2f}")
    
    # Test VWAP strategy
    mock_volume = [1000, 1500, 2000, 1800, 1200] * 2  # 10 periods
    vwap_strategy = optimizer.vwap_strategy(total_quantity, mock_volume, time_horizon)
    print(f"✅ VWAP: {len(vwap_strategy.slice_sizes)} slices")
    print(f"   Largest slice: {max(vwap_strategy.slice_sizes):.0f}")
    print(f"   Smallest slice: {min(vwap_strategy.slice_sizes):.0f}")
    
    # Test Implementation Shortfall
    is_strategy = optimizer.implementation_shortfall_strategy(total_quantity, time_horizon)
    print(f"✅ Implementation Shortfall: {len(is_strategy.slice_sizes)} slices")
    print(f"   First slice: {is_strategy.slice_sizes[0]:.0f}")
    print(f"   Last slice: {is_strategy.slice_sizes[-1]:.0f}")
    
    # Test Adaptive strategy
    market_conditions = {
        'volatility': 0.03,
        'avg_volume': 500000,
        'spread': 0.002
    }
    adaptive_strategy = optimizer.adaptive_strategy(total_quantity, time_horizon, market_conditions)
    print(f"✅ Adaptive: {len(adaptive_strategy.slice_sizes)} slices")
    print(f"   Strategy selected based on market conditions")

if __name__ == "__main__":
    main()
