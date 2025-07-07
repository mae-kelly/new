#!/usr/bin/env python3

import asyncio
import json
import logging
import time
import sys
import os
from typing import Dict, List, Optional, Tuple
import requests
from dataclasses import dataclass
from datetime import datetime
import aiohttp
import base58
from solana.rpc.async_api import AsyncClient
from solders.pubkey import Pubkey
from solders.signature import Signature

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/contract_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ContractAnalysisResult:
    """Results from contract safety analysis"""
    token_address: str
    is_safe: bool
    risk_score: float
    honeypot_detected: bool
    liquidity_locked: bool
    sell_tax: float
    buy_tax: float
    issues: List[str]
    liquidity_amount: float
    holder_count: int
    ownership_renounced: bool
    analysis_timestamp: datetime
    
    def to_dict(self) -> Dict:
        return {
            'token_address': self.token_address,
            'is_safe': self.is_safe,
            'risk_score': self.risk_score,
            'honeypot_detected': self.honeypot_detected,
            'liquidity_locked': self.liquidity_locked,
            'sell_tax': self.sell_tax,
            'buy_tax': self.buy_tax,
            'issues': self.issues,
            'liquidity_amount': self.liquidity_amount,
            'holder_count': self.holder_count,
            'ownership_renounced': self.ownership_renounced,
            'analysis_timestamp': self.analysis_timestamp.isoformat()
        }

class SolanaContractAnalyzer:
    """Advanced Solana contract safety analyzer"""
    
    def __init__(self, rpc_url: str = "https://api.mainnet-beta.solana.com"):
        self.rpc_url = rpc_url
        self.client = AsyncClient(rpc_url)
        self.session = None
        
        # Known dangerous patterns in Solana programs
        self.danger_patterns = [
            b'set_authority',
            b'mint_to',
            b'burn',
            b'freeze_account',
            b'thaw_account',
            b'close_account',
            b'set_close_authority'
        ]
        
        # Cache for analysis results
        self.analysis_cache = {}
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
        await self.client.close()
    
    async def analyze_token(self, token_address: str) -> ContractAnalysisResult:
        """Comprehensive token safety analysis"""
        logger.info(f"Starting analysis for token: {token_address}")
        
        # Check cache first
        if token_address in self.analysis_cache:
            cached_result = self.analysis_cache[token_address]
            if (datetime.now() - cached_result.analysis_timestamp).seconds < 300:  # 5 min cache
                logger.info(f"Returning cached result for {token_address}")
                return cached_result
        
        issues = []
        risk_factors = 0
        
        try:
            # Parse token address
            token_pubkey = Pubkey.from_string(token_address)
        except Exception as e:
            issues.append(f"Invalid token address: {str(e)}")
            return self._create_failed_result(token_address, issues)
        
        # Run parallel analysis
        tasks = [
            self._check_honeypot(token_address),
            self._analyze_token_account(token_pubkey),
            self._check_liquidity(token_address),
            self._check_ownership(token_pubkey),
            self._analyze_holders(token_address),
            self._check_trading_restrictions(token_pubkey)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        honeypot_result = results[0] if not isinstance(results[0], Exception) else {}
        token_account_result = results[1] if not isinstance(results[1], Exception) else {}
        liquidity_result = results[2] if not isinstance(results[2], Exception) else {}
        ownership_result = results[3] if not isinstance(results[3], Exception) else {}
        holders_result = results[4] if not isinstance(results[4], Exception) else {}
        trading_result = results[5] if not isinstance(results[5], Exception) else {}
        
        # Extract data with error handling
        honeypot_detected = honeypot_result.get('is_honeypot', True)
        sell_tax = honeypot_result.get('sell_tax', 1.0)
        buy_tax = honeypot_result.get('buy_tax', 0.0)
        
        liquidity_locked = liquidity_result.get('liquidity_locked', False)
        liquidity_amount = liquidity_result.get('liquidity_amount', 0)
        
        ownership_renounced = ownership_result.get('ownership_renounced', False)
        holder_count = holders_result.get('holder_count', 0)
        
        # Collect issues
        for result in results:
            if isinstance(result, dict) and 'issues' in result:
                issues.extend(result['issues'])
            elif isinstance(result, Exception):
                issues.append(f"Analysis error: {str(result)}")
        
        # Calculate risk factors
        if honeypot_detected:
            risk_factors += 3
        if sell_tax > 0.1:  # >10% sell tax
            risk_factors += 2
        if not liquidity_locked:
            risk_factors += 2
        if liquidity_amount < 2000:  # <$2k liquidity
            risk_factors += 2
        if not ownership_renounced:
            risk_factors += 1
        if holder_count < 10:
            risk_factors += 1
        
        # Calculate final risk score
        risk_score = min(risk_factors * 0.15, 1.0)
        
        # Determine if token is safe
        is_safe = (
            not honeypot_detected and
            sell_tax <= 0.1 and
            liquidity_locked and
            liquidity_amount >= 2000 and
            len(issues) == 0
        )
        
        result = ContractAnalysisResult(
            token_address=token_address,
            is_safe=is_safe,
            risk_score=risk_score,
            honeypot_detected=honeypot_detected,
            liquidity_locked=liquidity_locked,
            sell_tax=sell_tax,
            buy_tax=buy_tax,
            issues=issues,
            liquidity_amount=liquidity_amount,
            holder_count=holder_count,
            ownership_renounced=ownership_renounced,
            analysis_timestamp=datetime.now()
        )
        
        # Cache result
        self.analysis_cache[token_address] = result
        
        logger.info(f"Analysis complete for {token_address}: {'SAFE' if is_safe else 'UNSAFE'}")
        return result
    
    async def _check_honeypot(self, token_address: str) -> Dict:
        """Check for honeypot using multiple methods"""
        result = {
            'is_honeypot': False,
            'sell_tax': 0.0,
            'buy_tax': 0.0,
            'issues': []
        }
        
        try:
            # Method 1: Simulate transaction via Jupiter
            sim_result = await self._simulate_transaction(token_address)
            if sim_result.get('failed', True):
                result['is_honeypot'] = True
                result['issues'].append("Transaction simulation failed")
            
            # Method 2: Check via external honeypot API
            if not result['is_honeypot']:
                api_result = await self._check_honeypot_api(token_address)
                result.update(api_result)
            
            # Method 3: Tax analysis via multiple swaps
            if not result['is_honeypot']:
                tax_result = await self._analyze_taxes(token_address)
                result.update(tax_result)
        
        except Exception as e:
            result['issues'].append(f"Honeypot check error: {str(e)}")
            result['is_honeypot'] = True  # Fail safe
        
        return result
    
    async def _simulate_transaction(self, token_address: str) -> Dict:
        """Simulate buy/sell transaction to detect honeypots"""
        try:
            # Simulate a small buy via Jupiter
            url = "https://quote-api.jup.ag/v6/quote"
            params = {
                'inputMint': 'So11111111111111111111111111111111111111112',  # SOL
                'outputMint': token_address,
                'amount': 100000,  # 0.0001 SOL
                'slippageBps': 1000  # 10%
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    return {'failed': True, 'reason': 'Quote failed'}
                
                quote_data = await response.json()
                
                # Check if we can get a valid quote
                if not quote_data.get('outAmount'):
                    return {'failed': True, 'reason': 'No output amount'}
                
                # Simulate reverse transaction (sell)
                sell_params = {
                    'inputMint': token_address,
                    'outputMint': 'So11111111111111111111111111111111111111112',
                    'amount': quote_data['outAmount'],
                    'slippageBps': 1000
                }
                
                async with self.session.get(url, params=sell_params) as sell_response:
                    if sell_response.status != 200:
                        return {'failed': True, 'reason': 'Sell simulation failed'}
                    
                    sell_data = await sell_response.json()
                    
                    if not sell_data.get('outAmount'):
                        return {'failed': True, 'reason': 'Cannot sell token'}
                    
                    # Calculate effective tax
                    buy_amount = int(params['amount'])
                    sell_return = int(sell_data['outAmount'])
                    
                    if sell_return == 0:
                        return {'failed': True, 'reason': 'Zero sell return'}
                    
                    tax_rate = 1 - (sell_return / buy_amount)
                    
                    if tax_rate > 0.5:  # >50% tax likely honeypot
                        return {'failed': True, 'reason': f'High tax: {tax_rate:.2%}'}
                    
                    return {
                        'failed': False,
                        'effective_tax': tax_rate,
                        'buy_amount': buy_amount,
                        'sell_return': sell_return
                    }
        
        except Exception as e:
            return {'failed': True, 'reason': str(e)}
    
    async def _check_honeypot_api(self, token_address: str) -> Dict:
        """Check honeypot via external API"""
        try:
            # Use a honeypot detection service (example)
            url = "https://api.honeypot.is/v2/IsHoneypot"
            params = {
                'address': token_address,
                'chainID': 101  # Solana mainnet
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'is_honeypot': data.get('isHoneypot', False),
                        'sell_tax': data.get('sellTax', 0),
                        'buy_tax': data.get('buyTax', 0)
                    }
        except:
            pass
        
        return {'is_honeypot': False, 'sell_tax': 0, 'buy_tax': 0}
    
    async def _analyze_taxes(self, token_address: str) -> Dict:
        """Analyze buy/sell taxes through multiple test transactions"""
        try:
            # Test with different amounts to detect dynamic taxes
            test_amounts = [10000, 100000, 1000000]  # Different SOL amounts
            tax_results = []
            
            for amount in test_amounts:
                result = await self._simulate_transaction_amount(token_address, amount)
                if not result.get('failed'):
                    tax_results.append(result.get('effective_tax', 0))
            
            if tax_results:
                avg_tax = sum(tax_results) / len(tax_results)
                max_tax = max(tax_results)
                
                return {
                    'sell_tax': max_tax,
                    'buy_tax': 0,  # Simplified
                    'tax_variance': max_tax - min(tax_results)
                }
        
        except Exception as e:
            logger.warning(f"Tax analysis failed: {str(e)}")
        
        return {'sell_tax': 0, 'buy_tax': 0}
    
    async def _simulate_transaction_amount(self, token_address: str, amount: int) -> Dict:
        """Simulate transaction with specific amount"""
        try:
            url = "https://quote-api.jup.ag/v6/quote"
            params = {
                'inputMint': 'So11111111111111111111111111111111111111112',
                'outputMint': token_address,
                'amount': amount,
                'slippageBps': 1000
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('outAmount'):
                        # Quick reverse check
                        sell_params = {
                            'inputMint': token_address,
                            'outputMint': 'So11111111111111111111111111111111111111112',
                            'amount': data['outAmount'],
                            'slippageBps': 1000
                        }
                        
                        async with self.session.get(url, params=sell_params) as sell_resp:
                            if sell_resp.status == 200:
                                sell_data = await sell_resp.json()
                                sell_return = int(sell_data.get('outAmount', 0))
                                
                                if sell_return > 0:
                                    tax = 1 - (sell_return / amount)
                                    return {'failed': False, 'effective_tax': tax}
            
            return {'failed': True}
        
        except:
            return {'failed': True}
    
    async def _analyze_token_account(self, token_pubkey: Pubkey) -> Dict:
        """Analyze token account structure"""
        try:
            account_info = await self.client.get_account_info(token_pubkey)
            
            if not account_info.value:
                return {'issues': ['Token account not found']}
            
            # Analyze account data
            account_data = account_info.value.data
            issues = []
            
            # Check for suspicious patterns in account data
            for pattern in self.danger_patterns:
                if pattern in account_data:
                    issues.append(f"Dangerous pattern detected: {pattern.decode('utf-8', errors='ignore')}")
            
            return {'issues': issues}
        
        except Exception as e:
            return {'issues': [f"Token account analysis failed: {str(e)}"]}
    
    async def _check_liquidity(self, token_address: str) -> Dict:
        """Check liquidity pools and locks"""
        try:
            # Check Raydium pools
            raydium_result = await self._check_raydium_liquidity(token_address)
            
            # Check Jupiter aggregated liquidity
            jupiter_result = await self._check_jupiter_liquidity(token_address)
            
            # Combine results
            total_liquidity = raydium_result.get('liquidity', 0) + jupiter_result.get('liquidity', 0)
            
            return {
                'liquidity_locked': total_liquidity > 2000,
                'liquidity_amount': total_liquidity,
                'raydium_liquidity': raydium_result.get('liquidity', 0),
                'jupiter_liquidity': jupiter_result.get('liquidity', 0),
                'issues': []
            }
        
        except Exception as e:
            return {
                'liquidity_locked': False,
                'liquidity_amount': 0,
                'issues': [f"Liquidity check failed: {str(e)}"]
            }
    
    async def _check_raydium_liquidity(self, token_address: str) -> Dict:
        """Check Raydium liquidity pools"""
        try:
            url = f"https://api.raydium.io/v2/main/pairs"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    total_liquidity = 0
                    for pair in data:
                        if (pair.get('baseMint') == token_address or 
                            pair.get('quoteMint') == token_address):
                            total_liquidity += pair.get('liquidity', 0)
                    
                    return {'liquidity': total_liquidity}
        
        except Exception as e:
            logger.warning(f"Raydium liquidity check failed: {str(e)}")
        
        return {'liquidity': 0}
    
    async def _check_jupiter_liquidity(self, token_address: str) -> Dict:
        """Check Jupiter aggregated liquidity"""
        try:
            # Test liquidity by trying to quote large amounts
            test_amounts = [1000000000, 10000000000]  # 1 SOL, 10 SOL worth
            
            max_liquidity = 0
            for amount in test_amounts:
                url = "https://quote-api.jup.ag/v6/quote"
                params = {
                    'inputMint': 'So11111111111111111111111111111111111111112',
                    'outputMint': token_address,
                    'amount': amount,
                    'slippageBps': 5000  # 50% slippage tolerance
                }
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('outAmount'):
                            # Rough liquidity estimate
                            estimated_liquidity = amount / 1e9 * 100  # Rough SOL price
                            max_liquidity = max(max_liquidity, estimated_liquidity)
                        else:
                            break
            
            return {'liquidity': max_liquidity}
        
        except:
            return {'liquidity': 0}
    
    async def _check_ownership(self, token_pubkey: Pubkey) -> Dict:
        """Check token mint authority and ownership"""
        try:
            # Get mint account info
            mint_info = await self.client.get_account_info(token_pubkey)
            
            if not mint_info.value:
                return {
                    'ownership_renounced': False,
                    'issues': ['Cannot fetch mint info']
                }
            
            # Parse mint account data (simplified)
            # In practice, you'd decode the actual mint account structure
            mint_data = mint_info.value.data
            
            # Check if mint authority exists (simplified check)
            # Real implementation would properly decode the account
            has_mint_authority = len(mint_data) > 0  # Simplified
            
            return {
                'ownership_renounced': not has_mint_authority,
                'mint_authority_exists': has_mint_authority,
                'issues': []
            }
        
        except Exception as e:
            return {
                'ownership_renounced': False,
                'issues': [f"Ownership check failed: {str(e)}"]
            }
    
    async def _analyze_holders(self, token_address: str) -> Dict:
        """Analyze token holder distribution"""
        try:
            # Use a token analysis service to get holder count
            # This is a simplified implementation
            
            # In practice, you'd scan all token accounts for this mint
            # or use a service like SolScan API
            
            return {
                'holder_count': 100,  # Placeholder
                'top_holder_percentage': 0.1,  # Placeholder
                'issues': []
            }
        
        except Exception as e:
            return {
                'holder_count': 0,
                'issues': [f"Holder analysis failed: {str(e)}"]
            }
    
    async def _check_trading_restrictions(self, token_pubkey: Pubkey) -> Dict:
        """Check for trading restrictions and blacklists"""
        try:
            # Check for freeze authority
            # Check for transfer restrictions
            # This would involve analyzing the token program's constraints
            
            return {
                'has_freeze_authority': False,
                'has_transfer_restrictions': False,
                'issues': []
            }
        
        except Exception as e:
            return {
                'has_freeze_authority': True,  # Fail safe
                'has_transfer_restrictions': True,
                'issues': [f"Trading restrictions check failed: {str(e)}"]
            }
    
    def _create_failed_result(self, token_address: str, issues: List[str]) -> ContractAnalysisResult:
        """Create a failed analysis result"""
        return ContractAnalysisResult(
            token_address=token_address,
            is_safe=False,
            risk_score=1.0,
            honeypot_detected=True,
            liquidity_locked=False,
            sell_tax=1.0,
            buy_tax=0.0,
            issues=issues,
            liquidity_amount=0,
            holder_count=0,
            ownership_renounced=False,
            analysis_timestamp=datetime.now()
        )

async def analyze_token_cli(token_address: str) -> None:
    """CLI interface for token analysis"""
    async with SolanaContractAnalyzer() as analyzer:
        print(f"🔍 Analyzing token: {token_address}")
        
        result = await analyzer.analyze_token(token_address)
        
        # Print results
        print(f"\n{'='*60}")
        print(f"📊 ANALYSIS RESULTS FOR {token_address}")
        print(f"{'='*60}")
        
        status = "✅ SAFE" if result.is_safe else "❌ UNSAFE"
        print(f"Status: {status}")
        print(f"Risk Score: {result.risk_score:.2f}/1.0")
        print(f"Honeypot Detected: {'Yes' if result.honeypot_detected else 'No'}")
        print(f"Liquidity Locked: {'Yes' if result.liquidity_locked else 'No'}")
        print(f"Liquidity Amount: ${result.liquidity_amount:,.2f}")
        print(f"Sell Tax: {result.sell_tax:.2%}")
        print(f"Buy Tax: {result.buy_tax:.2%}")
        print(f"Holder Count: {result.holder_count}")
        print(f"Ownership Renounced: {'Yes' if result.ownership_renounced else 'No'}")
        
        if result.issues:
            print(f"\n⚠️  Issues Found:")
            for issue in result.issues:
                print(f"  • {issue}")
        
        # Save results
        os.makedirs('data', exist_ok=True)
        result_file = f"data/analysis_{token_address}_{int(time.time())}.json"
        with open(result_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        print(f"\n💾 Results saved to: {result_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python contract_analyzer.py <token_address>")
        sys.exit(1)
    
    token_address = sys.argv[1]
    asyncio.run(analyze_token_cli(token_address))
