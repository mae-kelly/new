#!/usr/bin/env python3
import os
import base58
from solana.keypair import Keypair
from solana.rpc.api import Client
from solders.pubkey import Pubkey

def setup_wallet():
    """Setup and verify wallet connection"""
    
    # Check for private key
    private_key = os.getenv('SOLANA_PRIVATE_KEY')
    if not private_key:
        print("❌ SOLANA_PRIVATE_KEY not found in environment")
        print("💡 Generate a new wallet or add your existing private key to .env")
        
        # Generate new wallet for demo
        new_keypair = Keypair()
        new_private_key = base58.b58encode(new_keypair.secret()).decode('ascii')
        
        print(f"\n🔑 Generated new wallet:")
        print(f"Public Key: {new_keypair.pubkey()}")
        print(f"Private Key: {new_private_key}")
        print(f"\n⚠️  Add this to your .env file: SOLANA_PRIVATE_KEY={new_private_key}")
        print("💰 Fund this wallet with SOL before trading!")
        return False
    
    try:
        # Test wallet connection
        keypair = Keypair.from_secret_key(private_key)
        client = Client("https://api.mainnet-beta.solana.com")
        
        # Get balance
        balance_response = client.get_balance(keypair.pubkey())
        balance_sol = balance_response.value / 1e9
        
        print(f"✅ Wallet connected successfully")
        print(f"📍 Address: {keypair.pubkey()}")
        print(f"💰 Balance: {balance_sol:.4f} SOL")
        
        if balance_sol < 0.01:
            print("⚠️  Low balance - add SOL for trading and fees")
        
        return True
        
    except Exception as e:
        print(f"❌ Wallet setup failed: {str(e)}")
        return False

if __name__ == "__main__":
    setup_wallet()
